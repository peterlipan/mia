import torch
import pandas as pd
import torch.nn.functional as F
from .metrics import compute_avg_metrics
import torch.distributed as dist
from einops import rearrange
from .losses import MultiTaskSupervisedContrast, MultiTaskSampleRelationLoss
from torch.utils.data import DataLoader
from datasets import AbideROIDataset, Transforms
from models import get_model
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, args, logger=None):

        self.csv_file = pd.read_csv(args.csv_path)
        self.transforms = Transforms()
        self.logger = logger
        self.args = args
    
    def split_datasets(self, train_pid, test_pid, args):
        train_csv = self.csv_file[self.csv_file['SUB_ID'].isin(train_pid)]
        test_csv = self.csv_file[self.csv_file['SUB_ID'].isin(test_pid)]
        self.train_dataset = AbideROIDataset(train_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
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
        
        n_classes = self.train_dataset.n_classes
        self.n_classes = n_classes
        args.n_classes = n_classes
        
        if args.rank == 0:
            self.test_dataset = AbideROIDataset(test_csv, args.data_root, atlas=args.atlas, task=args.task, n_views=args.n_views,
                                            transforms=self.transforms.test_transforms, cp=args.cp, cnp=args.cnp)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, 
                                          num_workers=args.workers, pin_memory=True, collate_fn=AbideROIDataset.collate_fn)
        else:
            self.test_loader = None
        
        step_per_epoch = len(self.train_dataset) // (args.batch_size * args.world_size)
        self.model = get_model(args).cuda()

        self.optimizer = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr, 
                                                              weight_decay=args.weight_decay)
        
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        self.cnp_criterion = MultiTaskSampleRelationLoss(batch_size=args.batch_size, world_size=args.world_size, 
                                                         num_phenotype=args.num_cnp).cuda()
        self.cp_criterion = MultiTaskSupervisedContrast(batch_size=args.batch_size, world_size=args.world_size, 
                                                        num_phenotype=args.num_cp).cuda()

        if args.scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup_epochs * step_per_epoch, 
                                                             args.epochs * step_per_epoch)
        else:
            self.scheduler = None
        
        if args.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.student, device_ids=[args.rank])


    def validate(self):
        self.model.eval()
        ground_truth = torch.Tensor().cuda()
        predictions = torch.Tensor().cuda()
        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                outputs = self.model(data['x'])
                pred = F.softmax(outputs.logits, dim=-1)
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
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) for k, v in data.items()}

                label_one_view = data['label'][:, 0].contiguous()
                label4cls = rearrange(data['label'], 'B V -> (B V)').contiguous()

                outputs = self.model(data['x'])
                cls_loss = self.ce(outputs.logits, label4cls)
                cp_loss = self.cp_criterion(outputs.cp_features, phenotypes=data['cp_label'], labels=label_one_view)
                cnp_loss = self.cnp_criterion(outputs.cnp_features, data['cnp_features'])
                loss = cls_loss + args.lambda_cp * cp_loss + args.lambda_cnp * cnp_loss

                self.optimizer.zero_grad()
                loss.backward()

                if dist.is_available() and dist.is_initialized():
                    for name, p in self.student.named_parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                            p.grad.data /= dist.get_world_size()
                        else:
                            print(f'None grad: {name}')

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                cur_iter += 1
                if cur_iter % 10 == 0:
                    if args.rank == 0:
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        metrics = self.validate()
                        print(f'Epoch: {epoch}, Iter: {cur_iter}, LR: {cur_lr}, Acc: {metrics['Accuracy']}')
                        if self.logger is not None:
                            self.logger.log({'test': metrics,
                                             'train': {
                                                 'cls_loss': cls_loss.item(), 
                                                 'cp_loss': cp_loss.item(),
                                                 'cnp_loss': cnp_loss.item(),}})

    def run(self, args):
        kf = KFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
        unique_patient = pd.unique(self.csv_file['SUB_ID'])
        for i, (train_idx, test_idx) in enumerate(kf.split(unique_patient)):
            if args.fold is not None and i != args.fold:
                continue
            train_patient_idx = unique_patient[train_idx]
            test_patient_idx = unique_patient[test_idx]
            self.split_datasets(train_patient_idx, test_patient_idx, args)
            self.train(args)
            if args.rank == 0:
                metrics = self.validate()
                print(f'Fold {i}: {metrics}')
