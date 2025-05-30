import os
import torch
import pandas as pd
import torch.nn.functional as F
from .metrics import compute_avg_metrics
import torch.distributed as dist
from einops import rearrange
from .pcgrad import PCGrad
from .losses import APheSCL, MultiviewCrossEntropy, MultiviewBCE
from torch.utils.data import DataLoader
from datasets import AbideROIDataset, Transforms, AdhdROIDataset
from models import get_model
from sklearn.model_selection import KFold, StratifiedKFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, args, logger=None):

        self.transforms = Transforms()
        self.logger = logger
        self.args = args
        self.result_csv_name = f"results_{args.dataset}_{args.task}.csv"
    
    def init_adhd_datasets(self, args):
        train_csv_path = os.path.join(args.csv_path, f"ADHD200_{args.atlas.lower()}_Training.csv")
        test_csv_path = os.path.join(args.csv_path, f"ADHD200_{args.atlas.lower()}_Testing.csv")
        train_csv = pd.read_csv(train_csv_path)
        test_csv = pd.read_csv(test_csv_path)
        ttpl_len = int(test_csv.shape[0] * args.ttpl_ratio)
        train_sample = train_csv.sample(n=ttpl_len, random_state=args.seed)
        ttpl_csv = pd.concat([train_sample, test_csv])
        self.train_dataset = AdhdROIDataset(train_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                            transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp, 
                                            task=args.task, filter='Both')
        # For test-time phenotype learning
        self.ttpl_dataset = AdhdROIDataset(ttpl_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                           transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp, 
                                            task=args.task, filter='Yes')
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            train_sampler = None

        
        args.num_cp = self.train_dataset.num_cp
        args.num_cnp = self.train_dataset.num_cnp

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                        drop_last=True, num_workers=args.workers, sampler=train_sampler, pin_memory=True,
                                        collate_fn=AdhdROIDataset.collate_fn)
        self.ttpl_loader = DataLoader(self.ttpl_dataset, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False, num_workers=args.workers, pin_memory=True,
                                        collate_fn=AdhdROIDataset.collate_fn)
        
        n_clasees = 1 if args.mixup else self.train_dataset.n_classes
        self.n_classes = n_clasees
        args.n_classes = n_clasees

        if args.rank == 0:
            self.test_dataset = AdhdROIDataset(test_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                              transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp, task=args.task)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                         num_workers=args.workers, pin_memory=True, collate_fn=AdhdROIDataset.collate_fn)
            self.val_loader = None
            print(f"Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}, TTPL: {len(self.ttpl_dataset)}")
        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_abide_datasets(self, args):
        train_csv = pd.read_csv(args.train_csv)
        test_csv = pd.read_csv(args.test_csv)
        val_csv = pd.read_csv(args.val_csv)
        ttpl_len = int(test_csv.shape[0] * args.ttpl_ratio)
        train_sample = train_csv.sample(n=ttpl_len, random_state=args.seed)
        ttpl_csv = pd.concat([train_sample, test_csv])
        self.train_dataset = AbideROIDataset(train_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
                                             transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp)
        string2index = self.train_dataset.string2index # ensure consistent string2index mapping
        self.ttpl_dataset = AbideROIDataset(ttpl_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
                                          transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp,
                                          string2index=string2index)
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            train_sampler = None
        
        args.num_cp = self.train_dataset.num_cp
        args.num_cnp = self.train_dataset.num_cnp

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                       drop_last=True, num_workers=args.workers, sampler=train_sampler, pin_memory=True,
                                       collate_fn=AbideROIDataset.collate_fn)
        self.ttpl_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=False,
                                       drop_last=False, num_workers=args.workers, pin_memory=True,
                                       collate_fn=AbideROIDataset.collate_fn)
        
        n_classes = 1 if args.mixup else self.train_dataset.n_classes
        self.n_classes = n_classes
        args.n_classes = n_classes
        
        if args.rank == 0:
            self.test_dataset = AbideROIDataset(test_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=1,
                                            transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp,
                                            string2index=string2index)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True, collate_fn=AbideROIDataset.collate_fn)
            
            self.val_dataset = AbideROIDataset(val_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=1,
                                            transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp,
                                            string2index=string2index)
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True, collate_fn=AbideROIDataset.collate_fn)
            print(f"Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}, TTPL: {len(self.ttpl_dataset)}")

        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_model(self, args, reload=False):
        
        step_per_epoch = len(self.train_dataset) // (args.batch_size * args.world_size)
        self.model = get_model(args)
        if reload:
            self.model.load_state_dict(torch.load(args.reload))
        self.model = self.model.cuda()

        opt = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr,
                                                              weight_decay=args.weight_decay)
        # self.optimizer = PCGrad(opt)
        self.optimizer = PCGrad(opt, temperature=args.temp_gd, decay_rate=args.temp_decay) if args.pcgrad else opt
        
        self.ce = MultiviewBCE().cuda() if args.mixup else MultiviewCrossEntropy().cuda()
        self.con = APheSCL(batch_size=args.batch_size, world_size=args.world_size, 
                            temperature=args.temp_con, alpha=args.alpha_con).cuda()

        if args.scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(opt, args.warmup_epochs * step_per_epoch, 
                                                             args.epochs * step_per_epoch)
        else:
            self.scheduler = None
        
        if args.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[args.rank], static_graph=True)

    def validate(self, loader):
        self.model.eval()
        ground_truth = torch.Tensor().cuda()
        predictions = torch.Tensor().cuda()
        with torch.no_grad():
            for data in loader:
                data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                outputs = self.model(data['x'])
                logits = outputs.logits.squeeze(1) if outputs.logits.dim() > 2 else outputs.logits # [B, 1, C] -> [B, C]
                pred = F.softmax(logits, dim=-1) if logits.size(-1) > 1 else F.sigmoid(logits) 
                ground_truth = torch.cat((ground_truth, data['label']))
                predictions = torch.cat((predictions, pred))
            
            metric = compute_avg_metrics(ground_truth, predictions, avg='micro')
        self.model.train()
        return metric
    
    def ema_update(self, alpha=0.99, global_step=0):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)

    def train(self, args):
        cur_iter = 0
        for epoch in range(args.epochs):
            self.model.train()
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                
                outputs = self.model(data['x']) 
                cls_loss = self.ce(outputs.logits, data['label'])
                con_loss = args.lambda_con * self.con(features=outputs.cp_features, labels=data['label'],
                                                      cat_phenotypes=data['cp_label'], cont_phenotypes=data['cnp_label'])
                loss = cls_loss + con_loss

                self.optimizer.zero_grad()
                conflict = 0.0
                accept_prob = 0.0
                if args.pcgrad:
                    self.optimizer.pc_backward(main_obj=cls_loss, aux_objs=[con_loss])
                    conflict = self.optimizer.conflict_intensity
                    accept_prob = self.optimizer.acceptance_prob
                else:
                    loss.backward()

                grad_norm = 0.0
                if dist.is_available() and dist.is_initialized():
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                            p.grad.data /= dist.get_world_size()
                            grad_norm += p.grad.data.norm(2).item() ** 2
                        else:
                            print(f'None grad: {name}')
                    grad_norm = grad_norm ** 0.5
                
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                cur_iter += 1
                if cur_iter % 10 == 1:
                    if args.rank == 0:
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        test_metrics = self.validate(self.test_loader)
                        val_metrics = self.validate(self.val_loader) if self.val_loader is not None else {'Accuracy': 0.0}
                        cur_temp = self.optimizer.cur_temp if args.pcgrad else 0
                        print(f'Epoch: {epoch}, Iter: {cur_iter}, LR: {cur_lr}, Acc: {test_metrics['Accuracy']}')
                        if self.logger is not None:
                            self.logger.log({'test': test_metrics, 'val': val_metrics,
                                             'train': {
                                                 'grad_norm': grad_norm,
                                                 'lr': cur_lr,
                                                'conflict': conflict,
                                                'accept_prob': accept_prob,
                                                 'cur_temp': cur_temp,
                                                'loss': loss.item(),
                                                 'cls_loss': cls_loss.item(), 
                                                 'con_loss': con_loss.item(),}})

    def test_time_phenotype_learning(self, args):
        self.model.train()
        # disable dropout
        dropout_modules = [module for module in self.model.modules() if isinstance(module,torch.nn.Dropout)]
        for module in dropout_modules:
            module.eval()
        opt = torch.optim.Adam(self.model.parameters(), lr=args.ttpl_lr)
        opt.zero_grad()
        for data in self.ttpl_loader:
            data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
            outputs = self.model(data['x'])
            con_loss = self.con(features=outputs.cp_features, labels=None,
                                cat_phenotypes=data['cp_label'], cont_phenotypes=data['cnp_label'])
            
            con_loss.backward() # gradient accumulation

        if dist.is_available() and dist.is_initialized():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= dist.get_world_size()  # Average gradients across all GPUs

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        opt.step()

    def run(self, args):
        if 'ABIDE' in args.dataset:
            self.init_abide_datasets(args)
        else:
            self.init_adhd_datasets(args)
        self.init_model(args)
        self.train(args)
        if args.rank == 0:
            metrics = self.validate(self.test_loader)
            print(f'Final: {metrics}')
            self.save_model(args, metrics)
            self.save_results(args, metrics)
        self.test_time_phenotype_learning(args)
        if args.rank == 0:
            metrics = self.validate(self.test_loader)
            print(f'Final: {metrics}')
            self.save_results(args, metrics, ttpl=True)
            self.save_model(args, metrics, ttpl=True)

    def save_model(self, args, performance, ttpl=False):
        state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        name = f"{args.dataset}_{args.atlas}_{args.task}_AUC_{performance['AUC']:.4f}"
        if ttpl:
            name += "_TTPL"
        name += ".pth"
        save_path = os.path.join(args.checkpoints, name)
        torch.save(state_dict, save_path)
    
    def load_model(self, args):
        model_prefix = f"{args.dataset}_{args.atlas}_{args.task}"
        candidates = [f for f in os.listdir(args.checkpoints) if f.startswith(model_prefix)]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No pretrained model for condition {model_prefix}")
        candidates.sort()
        model_path = os.path.join(args.checkpoints, candidates[-1])
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def save_results(self, args, metrics, ttpl=False):
        cols = ['Model', 'Dataset', 'Atlas', 'Task', 'Seed', 'Omega'] + list(metrics.keys())
        if not os.path.exists(self.result_csv_name):
            results = pd.DataFrame(columns=cols)
        else:
            results = pd.read_csv(self.result_csv_name)
            assert set(results.columns) == set(cols), "Columns mismatch"
        model_name = args.model if not ttpl else f'{args.model} w/ TTPL'
        row = [model_name, args.dataset, args.atlas, args.task, args.seed, args.ttpl_ratio] + [metrics[key] for key in metrics.keys()]
        results = results._append(pd.Series(row, index=cols), ignore_index=True)
        results.to_csv(self.result_csv_name, index=False)
    
    def inference(self, args):
        print(f"Running inference for {args.model} on {args.dataset} dataset")
        if 'ABIDE' in args.dataset:
            self.init_abide_datasets(args)
        else:
            self.init_adhd_datasets(args)
        self.init_model(args, reload=True)
        if args.rank == 0:
            self.save_features()
            # metrics = self.validate(self.test_loader)
            # print(f'Final: {metrics}')
            # self.save_results(args, metrics=metrics, ttpl=False)

    def run_ttpl(self, args):
        if 'ABIDE' in args.dataset:
            self.init_abide_datasets(args)
        else:
            self.init_adhd_datasets(args)
        args.dropout = 0.0
        self.init_model(args, reload=True)
        self.test_time_phenotype_learning(args)
        if args.rank == 0:
            metrics = self.validate(self.test_loader)
            print(f'Final: {metrics}')
            self.save_results(args, metrics, ttpl=True)

    def save_features(self):
        self.model.eval()
        samples = {
            'features': [],
            'probs': [],
            'labels': [],
            'phenotypes': []
        }
        filename = f"Features_{self.args.dataset}_{self.args.atlas}_{self.args.model}_{self.args.fusion}Fusion.pt"
        save_path = os.path.join(self.args.results, filename)
        
        with torch.no_grad():
            for data in self.test_loader:
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
                outputs = self.model(data['x'])
                cp = data['cp_label'].cpu()
                cnp = data['cnp_label'].cpu()
                phe = torch.cat((cp, cnp), dim=-1)
                
                # Collect entire batch tensors
                samples['features'].append(outputs.features.cpu())
                samples['probs'].append(F.sigmoid(outputs.logits).cpu())
                samples['labels'].append(data['label'].cpu())
                samples['phenotypes'].append(phe)
            
            for data in self.val_loader:
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
                outputs = self.model(data['x'])
                cp = data['cp_label'].cpu()
                cnp = data['cnp_label'].cpu()
                phe = torch.cat((cp, cnp), dim=-1)
                
                # Collect entire batch tensors
                samples['features'].append(outputs.features.cpu())
                samples['probs'].append(F.sigmoid(outputs.logits).cpu())
                samples['labels'].append(data['label'].cpu())
                samples['phenotypes'].append(phe)

            # Concatenate all batches along the first dimension
            samples = {k: torch.cat(v, dim=0) for k, v in samples.items()}
        
        torch.save(samples, save_path)
        print(f"Features saved at {save_path}")