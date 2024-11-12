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
from einops import rearrange
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
            print(loss)

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
                if cur_iter % 50 == 0:
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


def expectation_step(encoder, aggregator, classifier, train_loader, optimizer, logger=None):
    # Expectation step (for one epoch)
    # Freeze the classifier and train the encoder of frames
    # to extimate the distribution of frame-level features
    encoder.train()
    aggregator.eval()
    classifier.eval()

    criteria = nn.CrossEntropyLoss().cuda()
    start = time.time()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(0)
    for i, (img, label) in enumerate(train_loader):
        img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
        features = encoder(img)
        with torch.no_grad():
            # features: [B, C] -> [B, 1, C]
            features = aggregator(features.unsqueeze(1))
            logits = classifier(features)
        loss = criteria(logits, label)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        print(f"Expectation step: {i} / {len(train_loader)}")
        if i % 500 == 0 and logger is not None:
            logger.log({'E Step': {'loss': loss.item()}})

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
    print(f'Expectation step finished in {time.time() - start:.4f} sec')

    return encoder


def maximization_step(encoder, aggregator, classifier, train_loader, optimizer, logger=None):
    # Maximization step (for one epoch)
    # Freeze the encoder and train the aggregator and classifier
    encoder.eval()
    aggregator.train()
    classifier.train()

    criteria = nn.CrossEntropyLoss().cuda()
    start = time.time()

    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(0)
    for i, (img, label) in enumerate(train_loader):
        img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
        # img: [B, T, C, D, H, W] -> [BxT, C, D, H, W]
        bs = img.size(0)
        img = rearrange(img, 'B T C D H W -> (B T) C D H W')

        with torch.no_grad():
            features = encoder(img)
            # features: [BxT, C] -> [B, T, C]
            features = rearrange(features, '(B T) C -> B T C', B=bs)

        fmri_features = aggregator(features)
        logits = classifier(fmri_features)
        loss = criteria(logits, label)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        print(f"Maximization step: {i} / {len(train_loader)}")

        if i % 10 == 0 and logger is not None:
            logger.log({'M Step': {'loss': loss.item()}})
    print(f'Maximization step finished in {time.time() - start:.4f} sec')
    return aggregator, classifier
    

def validate(dataloader, encoder, aggregator, classifier):
    encoder.eval()
    aggregator.eval()
    classifier.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for step, (img, label) in enumerate(dataloader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            bs = img.size(0)
            # img [B, T, C, D, H, W] -> [BxT, C, D, H, W] and feed into the frame encoder
            img = rearrange(img, 'B T C D H W -> (B T) C D H W')
            features = encoder(img)
            # features: [BxT, C] -> [B, T, C]
            features = rearrange(features, '(B T) C -> B T C', B=bs)
            features = aggregator(features)
            # features: [B, C]
            logits = classifier(features)

            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))
            if step > 30:
                break

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='micro')
    encoder.train()
    aggregator.train()
    classifier.train()
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def iterative_training(loaders, models, optimizers, args, logger):
    train_frame_loader, train_fmri_loader, test_fmri_loader = loaders
    encoder, aggregator, classifier = models
    e_optimizer, m_optimizer = optimizers

    for epoch in range(args.epochs):
        encoder = expectation_step(encoder, aggregator.module, classifier.module, train_frame_loader, e_optimizer, logger)
        if os.path.exists(os.path.join(args.checkpoints, f'encoder_{epoch - 1}.pth')):
            os.remove(os.path.join(args.checkpoints, f'encoder_{epoch - 1}.pth'))
        torch.save(encoder.module.state_dict(), os.path.join(args.checkpoints, f'encoder_{epoch}.pth'))

        aggregator, classifier = maximization_step(encoder.module, aggregator, classifier, train_fmri_loader, m_optimizer, logger)
        
        if os.path.exists(os.path.join(args.checkpoints, f'aggregator_{epoch - 1}.pth')):
            os.remove(os.path.join(args.checkpoints, f'aggregator_{epoch - 1}.pth'))
        torch.save(aggregator.module.state_dict(), os.path.join(args.checkpoints, f'aggregator_{epoch}.pth'))
        if os.path.exists(os.path.join(args.checkpoints, f'classifier_{epoch - 1}.pth')):
            os.remove(os.path.join(args.checkpoints, f'classifier_{epoch - 1}.pth'))
        torch.save(classifier.module.state_dict(), os.path.join(args.checkpoints, f'classifier_{epoch}.pth'))
        if args.rank == 0:
            test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
                test_fmri_loader, encoder, aggregator, classifier)
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
                                     'Kappa': test_kappa}}, )
            print(f'Epoch: {epoch} finished')


def direct_validate(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for step, (img, label) in enumerate(dataloader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            
            logits = model(img)

            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='micro')
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def direct_training(loaders, model, optimizer, scheduler, args, logger):
    train_fmri_loader, test_fmri_loader = loaders

    criteria = nn.CrossEntropyLoss().cuda()

    model.train()
    
    cur_iter = 0
    
    for epoch in range(args.epochs):
        for img, label in train_fmri_loader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()

            logits = model(img)
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
            if cur_iter % 10 == 1:
                if args.rank == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = direct_validate(test_fmri_loader, model)
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
                                              'lr': cur_lr}}, )
