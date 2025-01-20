import os
import torch
import wandb
import time
import shutil
import numpy as np
import pandas as pd
from torch import nn
from einops import rearrange
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from .metrics import compute_avg_metrics
from .losses import MultiTaskSampleRelationLoss


def validate(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for step, (img, label, _, _) in enumerate(dataloader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            
            logits, *_ = model(img)

            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='micro')
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def train(loaders, model, optimizer, scheduler, args, logger):
    train_fmri_loader, test_fmri_loader = loaders

    criteria = nn.CrossEntropyLoss().cuda()
    con_criteria = MultiTaskSupervisedContrast(batch_size=args.batch_size, world_size=args.world_size, num_phenotype=args.num_cp).cuda()
    cnp_criteria = MultiTaskSampleRelationLoss(batch_size=args.batch_size, world_size=args.world_size, num_phenotype=args.num_cnp).cuda()

    model.train()
    
    cur_iter = 0
    
    for epoch in range(args.epochs):
        # necessary to set the epoch for the sampler to shuffle the data
        if isinstance(train_fmri_loader.sampler, DistributedSampler):
            train_fmri_loader.sampler.set_epoch(epoch)
        for img, label, cnp_fea, cp_fea in train_fmri_loader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            cnp_fea, cp_fea = cnp_fea.cuda(non_blocking=True), cp_fea.cuda(non_blocking=True)

            # [B, V] -> [B V,]
            con_label = label[:, 0].contiguous() # [B, V] -> [B,]
            label = rearrange(label, 'B V -> (B V)').contiguous() # [B, V] -> [B V,]

            logits, con_fea, fea = model(img)
            # phenotypes = torch.cat((cnp_fea, cp_fea), dim=-1).contiguous() # [B, V, K] -> [B, K]
            cls_loss = criteria(logits, label)
            con_loss = args.lambda_con * con_criteria(con_fea, phenotypes=cp_fea, labels=con_label) # TODO: add cnp
            cnp_loss = args.lambda_cnp * cnp_criteria(fea, cnp_fea)
            loss = cls_loss + con_loss + cnp_loss

            if args.rank == 0:
                cls_loss_val = cls_loss.item()
                con_loss_val = con_loss.item()
                cnp_loss_val = cnp_loss.item()
                cur_temp = optimizer.cur_temp if hasattr(optimizer, 'cur_temp') else 0
                train_loss = cls_loss_val + con_loss_val + cnp_loss_val

            optimizer.zero_grad()
            optimizer.pc_backward(main_obj=cls_loss, aux_objs=[con_loss, cnp_loss])
            # loss.backward()

            # Synchronize gradients across all processes
            if dist.is_available() and dist.is_initialized():
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # Sum gradients
                        p.grad.data /= dist.get_world_size() 
                    else:
                        print(f'None grad: {name}')
                
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            cur_iter += 1
            if cur_iter % 10 == 1:
                if args.rank == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(test_fmri_loader, model)
                    print(f"Epoch: {epoch} / {args.epochs} || Iter: {cur_iter} || Test Sens: {test_sens} || Test Spec: {test_spec} || Test ACC: {test_acc}")
                    if logger is not None:
                        logger.log({'test': {'Accuracy': test_acc,
                                            'F1 score': test_f1,
                                            'AUC': test_auc,
                                            'AP': test_ap,
                                            'Balanced Accuracy': test_bac,
                                            'Sensitivity': test_sens,
                                            'Specificity': test_spec,
                                            'Precision': test_prec,
                                            'MCC': test_mcc,
                                            'Kappa': test_kappa},
                                    'train': {'loss': train_loss,
                                            'cls_loss': cls_loss_val,
                                            'con_loss': con_loss_val,
                                            'cnp_loss': cnp_loss_val,
                                            'cur_temp': cur_temp,
                                            'lr': cur_lr}}, )
    return model


def test_time_train(dataloader, model, optimizer, args):
    con_criteria = MultiTaskSupervisedContrast(batch_size=args.batch_size, world_size=args.world_size, num_phenotype=args.num_cp).cuda()
    cnp_criteria = MultiTaskSampleRelationLoss(batch_size=args.batch_size, world_size=args.world_size, num_phenotype=args.num_cnp).cuda()
    model.train()
    for img, _, cnp_fea, cp_fea in dataloader:
        img = img.cuda(non_blocking=True)
        cnp_fea, cp_fea = cnp_fea.cuda(non_blocking=True), cp_fea.cuda(non_blocking=True)

        _, con_fea, fea = model(img)
        # phenotypes = torch.cat((cnp_fea, cp_fea), dim=-1).contiguous() # [B, V, K] -> [B, K]
        con_loss = args.lambda_con * con_criteria(con_fea, phenotypes=cp_fea) # TODO: add cnp
        cnp_loss = args.lambda_cnp * cnp_criteria(fea, cnp_fea)
        loss = (con_loss + cnp_loss) / img.size(0) # average loss per sample

        loss.backward() # gradient accumulation

    # Synchronize gradients across all processes
    if dist.is_available() and dist.is_initialized():
        for name, p in model.named_parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # Sum gradients
                p.grad.data /= dist.get_world_size()
            else:
                print(f'None grad: {name}')
    
    optimizer.step()
    
    return model
