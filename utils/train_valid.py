import os
import torch
import wandb
import time
import numpy as np
import pandas as pd
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from .metrics import compute_avg_metrics


def train(dataloaders, model, optimizer, scheduler, args, logger):

    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()

    criteria = nn.CrossEntropyLoss().cuda()

    cur_iter = 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, label) in enumerate(train_loader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            _, logits = model(img)

            # classification loss
            loss = criteria(logits, label)

            if args.rank == 0:
                train_loss = loss.item()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 50 == 1:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
                        test_loader, model, args.eval_steps)
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
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss), end='', flush=True)

    # TODO: save the model


def validate(dataloader, model, max_steps=100):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for step, (img, label) in enumerate(dataloader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            _, logits = model(img)
            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

            # for faster evaluation
            if step >= max_steps:
                break      

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='macro')
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa

