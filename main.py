# ------------------------------------------------------------------------
# Main script to commence baseline experiments on WEAR dataset
# ------------------------------------------------------------------------
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import pandas as pd
import numpy as np
import neptune
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from inertial_baseline.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.data_utils import label_dict
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from camera_baseline.actionformer.main import run_actionformer
from camera_baseline.tridet.main import run_tridet
from camera_baseline.temporal.main import run_temporal

def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="nishil007/Wear",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzlhZTIzYi1lM2FlLTRkZGEtOTE1Mi1iOTc2ZTA5ZDg3OTQifQ==",
        )
    else:
        run = None

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['devices'] = [args.gpu]

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    formatted_ts = ts.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', config['name'] , str(formatted_ts))
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt')) #to save all training logs

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush() #to save specific configuration which being used
    
    if args.neptune:
        run['eval_type'] = args.eval_type
        run['config_name'] = args.config
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
        run['params'] = config

    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))
    all_v_pred_post = np.array([])
    all_v_mAP_post = np.empty((0, len(config['dataset']['tiou_thresholds'])))
        
    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            anno_file = json.load(f)['database']
            train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
            val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i) #ex: 'split_0'
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_file'] = anno_split #ex: 'data/wear/annotations/wear_split_1.json'

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_mAP_post, v_preds, v_preds_post, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_mAP_post, v_preds, v_preds_post, v_gt = run_actionformer(config, log_dir, args.ckpt_freq, args.resume, rng_generator, run, i)
        elif config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_mAP_post, v_preds, v_preds_post, v_gt = run_tridet(config, log_dir, args.ckpt_freq, args.resume, rng_generator, run, i)
        elif config['name'] == 'temporal':
            t_losses, v_losses, v_mAP, v_mAP_post, v_preds, v_preds_post, v_gt = run_temporal(config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)

        # unprocessed results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true') #shape: (19,19)
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1) #shape (19,)
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1)
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1)
        
        # postprocessed results
        conf_mat_post = confusion_matrix(v_gt, v_preds_post, normalize='true')
        v_acc_post = conf_mat_post.diagonal()/conf_mat_post.sum(axis=1)
        v_prec_post = precision_score(v_gt, v_preds_post, average=None, zero_division=1)
        v_rec_post = recall_score(v_gt, v_preds_post, average=None, zero_division=1)
        v_f1_post = f1_score(v_gt, v_preds_post, average=None, zero_division=1)

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.mean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.mean(v_losses))
        block4 = ''
        block4 = 'SCORES NO POSTPROCESSING'
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.mean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.mean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.mean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.mean(v_f1) * 100)
        block6 = ''
        block6 = 'SCORES WITH POSTPROCESSING'
        block6  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(v_mAP_post) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP_post):
            block6 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block7 = ''
        block7  += '\t\tAcc {:>4.2f} (%)'.format(np.mean(v_acc_post) * 100)
        block7  += ' Prec {:>4.2f} (%)'.format(np.mean(v_prec_post) * 100)
        block7  += ' Rec {:>4.2f} (%)'.format(np.mean(v_rec_post) * 100)
        block7  += ' F1 {:>4.2f} (%)\n'.format(np.mean(v_f1_post) * 100)

        print('\n'.join([block1, block2, block3, block4, block5, block6, block7]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_mAP_post = np.append(all_v_mAP_post, v_mAP_post[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)
        all_v_pred_post = np.append(all_v_pred_post, v_preds_post)

        # save unprocessed confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=list(label_dict.keys()))
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (unprocessed)')
        plt.savefig(os.path.join(log_dir, name + '_unprocessed.png'))
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_unprocessed')

        # save postprocessed confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_post, display_labels=list(label_dict.keys()))
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (postprocessed)')
        plt.savefig(os.path.join(log_dir, name + '.png'))
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_postprocessed')

    # final unprocessed results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true') #shape: (19,19)
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1)
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1)
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1)

    # final postprocessed results across all splits
    conf_mat_post = confusion_matrix(all_v_gt, all_v_pred_post, normalize='true')
    v_acc_post = conf_mat_post.diagonal()/conf_mat_post.sum(axis=1)
    v_prec_post = precision_score(all_v_gt, all_v_pred_post, average=None, zero_division=1)
    v_rec_post = recall_score(all_v_gt, all_v_pred_post, average=None, zero_division=1)
    v_f1_post = f1_score(all_v_gt, all_v_pred_post, average=None, zero_division=1)

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2 = 'SCORES NO POSTPROCESSING'
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.mean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.mean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.mean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.mean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.mean(v_f1) * 100)
    block3 = 'SCORES WITH POSTPROCESSING'
    block3  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(all_v_mAP_post) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP_post.T):
        block3 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.mean(tiou_mAP)*100)
    block3  += '\n\t\tAcc {:>4.2f} (%)'.format(np.mean(v_acc_post) * 100)
    block3  += ' Prec {:>4.2f} (%)'.format(np.mean(v_prec_post) * 100)
    block3  += ' Rec {:>4.2f} (%)'.format(np.mean(v_rec_post) * 100)
    block3  += ' F1 {:>4.2f} (%)'.format(np.mean(v_f1_post) * 100)
    
    print('\n'.join([block1, block2, block3]))

    # save final unprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (unprocessed)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=list(label_dict.keys()))
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_unprocessed.png'))
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_unprocessed.png')), name='all')

    # save final postprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Raw Total (postprocessed)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_post, display_labels=list(label_dict.keys()))
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_postprocessed.png'))
    if run is not None:
        run['conf_raw_matrices'].append(File(os.path.join(log_dir, 'all_postprocessed.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.mean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.mean(tiou_mAP))
        run['final_accuracy'] = np.mean(v_acc)
        run['final_precision'] = (np.mean(v_prec))
        run['final_recall'] = (np.mean(v_rec))
        run['final_f1'] = (np.mean(v_f1))

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/120_frames_60_stride/tridet_combined.yaml')
    parser.add_argument('--eval_type', default='split')
    parser.add_argument('--neptune', default=False, type=bool) 
    parser.add_argument('--seed', default=1, type=int)        
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)  

