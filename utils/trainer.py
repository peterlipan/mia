import os
import torch
import pandas as pd
import torch.nn.functional as F
from .metrics import compute_avg_metrics
import torch.distributed as dist
from einops import rearrange
from .pcgrad import PCGrad
from .losses import APheSCL, MultiviewCrossEntropy
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
    
    def init_adhd_datasets(self, args):
        train_csv = pd.read_csv(args.train_csv)
        test_csv = pd.read_csv(args.test_csv)
        self.train_dataset = AdhdROIDataset(train_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                            transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp, task=args.task)
        # For test-time phenotype learning
        self.ttpl_dataset = AdhdROIDataset(test_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                           transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp, task=args.task)
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
        
        n_clasees = self.train_dataset.n_classes
        self.n_classes = n_clasees
        args.n_classes = n_clasees

        if args.rank == 0:
            self.test_dataset = AdhdROIDataset(test_csv, args.data_root, atlas=args.atlas, n_views=args.n_views,
                                              transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp, task=args.task)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                         num_workers=args.workers, pin_memory=True, collate_fn=AdhdROIDataset.collate_fn)
            self.val_loader = None
        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_abide_datasets(self, args):
        train_csv = pd.read_csv(args.train_csv)
        test_csv = pd.read_csv(args.test_csv)
        val_csv = pd.read_csv(args.val_csv)
        self.train_dataset = AbideROIDataset(train_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
                                          transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp)
        self.ttpl_dataset = AbideROIDataset(train_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
                                          transforms=self.transforms.train_transforms, cp=args.cp, cnp=args.cnp)
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
        
        n_classes = self.train_dataset.n_classes
        self.n_classes = n_classes
        args.n_classes = n_classes
        
        if args.rank == 0:
            self.test_dataset = AbideROIDataset(test_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=1,
                                            transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True, collate_fn=AbideROIDataset.collate_fn)
            
            self.val_dataset = AbideROIDataset(val_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=1,
                                            transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp)
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True, collate_fn=AbideROIDataset.collate_fn)
        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_model(self, args):
        
        step_per_epoch = len(self.train_dataset) // (args.batch_size * args.world_size)
        self.model = get_model(args).cuda()

        opt = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr,
                                                              weight_decay=args.weight_decay)
        # self.optimizer = PCGrad(opt)
        self.optimizer = PCGrad(opt, temperature=args.temp_gd, decay_rate=args.temp_decay) if args.pcgrad else opt
        
        self.ce = MultiviewCrossEntropy().cuda()
        self.con = APheSCL(batch_size=args.batch_size, world_size=args.world_size, temperature=args.temp_con).cuda()

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
                pred = F.softmax(outputs.logits.squeeze(1), dim=-1) # [B, 1, C] -> [B, C]
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

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
        if args.rank == 0:
            self.save_model(args)

    def test_time_phenotype_learning(self, args):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=args.ttpl, weight_decay=args.weight_decay)
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
                else:
                    print(f'None grad: {name}')
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        opt.step()
        
        if args.rank == 0:
            performance = self.validate(self.test_loader)
            print(f'Test time training: {performance}')

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
        self.test_time_phenotype_learning(args)

    def save_model(self, args):
        state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        performance = self.validate(self.test_loader)
        save_path = os.path.join(args.checkpoints, f"{args.dataset}_{args.atlas}_{args.task}_AUC_{performance['AUC']:.4f}_.pth")
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
