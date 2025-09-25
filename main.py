import os
import argparse
import numpy as np
import random
import sys
import time
# from tqdm import tqdm
# import ipdb
# import pickle

import torch
import torch.nn as nn
from Utils.utils import set_log,set_seed
from Configs.builder import get_configs
#<SETTINGS>
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
#<INITIAL_ARGUEMENTS>
parser = argparse.ArgumentParser(description="Uneven Event Modeling")
    #DATASET
parser.add_argument(
    '-d', '--dataset_name', default='act', type=str, metavar='DATASET', help='dataset name',choices=['tvr', 'act']
)
    #GPU
parser.add_argument(
    '--gpu', default = '1', type = str, help = 'specify gpu device'
)
    #EVALUATION
parser.add_argument('--eval', action='store_true')
    #LOAD CHECKPOINT
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()

def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()

    model.train()

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                    unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):

        batch = gpu(batch)

        optimizer.zero_grad()

        input_list = model(batch)

        loss = criterion(input_list, batch)
        
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.cpu().item())

        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss))

    return loss_meter.avg

def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val, loss_meter, logger):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'
                
    logger.info('==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3], best_val[4]))
    logger.info('==========================================================================================================')
        
    return val_meter, best_val, es

def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    
    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('==========================================================================================================')



def main():
    cfg = get_configs(args.dataset_name)
    
    # Set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info(f'Uneven Event Modeling: {cfg['dataset_name']}')


    # Set seed
    set_seed(cfg['seed'])
    logger.info(f'set seed: {cfg['seed']}')
    
    #Use device_ids to load Parralel GPU if need it
    device_ids = range(torch.cuda.device_count())
    logger.info(f'used gpu: {args.gpu}') 
    
    # Log Hyperparameters
    logger.info('Hyper Parameter ......')
    logger.info(cfg)




    
if __name__ == "__main__":
    main()