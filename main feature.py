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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.data_utils import label_dict
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt



import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight
import torch
from inertial_baseline.AttendAndDiscriminate import AttendAndDiscriminate
from inertial_baseline.DeepConvLSTM import DeepConvLSTM
from utils.data_utils import convert_samples_to_segments, label_dict, unwindow_inertial_data
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.torch_utils import init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from camera_baseline.actionformer.libs.utils.metrics import ANETdetection
from utils.os_utils import mkdir_if_missing
from scipy.signal import medfilt



def main(args):
    
    run = None

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['devices'] = [args.gpu]

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    formatted_ts = ts.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', config['name'] ,'_feature_', str(formatted_ts))
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

        extracted_features = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, run, i)

        folder_path = os.path.join(os.getcwd(), 'extracted feature', name )
        os.makedirs(folder_path, exist_ok=True)

        for key, value in extracted_features.items():
            file_path = os.path.join(folder_path, f"{key}.npy")  # File path for saving the tensor
            
            value = torch.cat(value, dim=0)
            # Move the tensor to CPU before saving
            cpu_tensor = value.cpu()
            
            # Convert the tensor to a NumPy array and save it
            np_array = cpu_tensor.numpy()
            np.save(file_path, np_array)
        
        del extracted_features




    print("ALL FINISHED")


def run_inertial_network(train_sbjs, val_sbjs, cfg, ckpt_folder, ckpt_freq, resume, run, iter):
    split_name = cfg['dataset']['json_file'].split('/')[-1].split('.')[0]
    
    train_data = {}
    val_data = {}
    # load train and val inertial data
    #train_data, val_data = np.empty((0, cfg['dataset']['input_dim'] + 2)), np.empty((0, cfg['dataset']['input_dim'] + 2))
    for t_sbj in train_sbjs:
        t_data = pd.read_csv(os.path.join(cfg['dataset']['sens_folder'], t_sbj + '.csv'), index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
        train_data[t_sbj] = t_data
    for v_sbj in val_sbjs:
        v_data  = pd.read_csv(os.path.join(cfg['dataset']['sens_folder'], v_sbj + '.csv'), index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
        val_data[v_sbj] = v_data

    train_dataset = {}
    test_dataset = {}
    train_loader = {}
    test_loader = {}
    # define inertial datasets
    for t_sbj in train_sbjs:
        train_dataset[t_sbj] = InertialDataset(train_data[t_sbj] , cfg['dataset']['window_size'], cfg['dataset']['window_overlap'], cfg['dataset']['include_null'])
        train_loader[t_sbj] = DataLoader(train_dataset[t_sbj], cfg['loader']['batch_size'], shuffle=False, num_workers=4)
    for v_sbj in val_sbjs:    
        test_dataset[v_sbj]  = InertialDataset(val_data[v_sbj], cfg['dataset']['window_size'], cfg['dataset']['window_overlap'], cfg['dataset']['include_null'])
        test_loader[v_sbj] = DataLoader(test_dataset[v_sbj], cfg['loader']['batch_size'], shuffle=False, num_workers=4)

    # define dataloaders
    #train_loader = DataLoader(train_dataset, cfg['loader']['batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    #test_loader = DataLoader(test_dataset, cfg['loader']['batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    
    # define network
    if cfg['name'] == 'deepconvlstm':
        net = DeepConvLSTM(
            12, 19, 50,
            cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'], 
            cfg['model']['lstm_units'], cfg['model']['lstm_layers'], cfg['model']['dropout'], feature_extract= 'conv'
            )
    elif cfg['name'] == 'attendanddiscriminate':
        net = AttendAndDiscriminate(
            train_dataset.channels, train_dataset.classes, cfg['model']['hidden_dim'], cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'], cfg['model']['enc_layers'], cfg['model']['enc_is_bidirectional'], cfg['model']['dropout'], cfg['model']['dropout_rnn'], cfg['model']['dropout_cls'], cfg['model']['activation'], cfg['model']['sa_div']
            )

    if iter == 0:
        path = r"C:\Users\nbala\HAR_Project\wear\logs\deepconvlstm\2023-06-19_16-50-49\ckpts\epoch_300_wear_split_1.pth.tar"    
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
    elif iter == 1:
        path = r"C:\Users\nbala\HAR_Project\wear\logs\deepconvlstm\2023-06-19_16-50-49\ckpts\epoch_300_wear_split_2.pth.tar"   
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
    else:
        path = r"C:\Users\nbala\HAR_Project\wear\logs\deepconvlstm\2023-06-19_16-50-49\ckpts\epoch_300_wear_split_3.pth.tar"   
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))

    #checkpoint_split_2['state_dict']['conv1.weight']

    #conv1_weight = (checkpoint_split_1['state_dict']['conv1.weight']+ checkpoint_split_2['state_dict']['conv1.weight'] + checkpoint_split_3['state_dict']['conv1.weight'])/3
    #conv2_weight = (checkpoint_split_1['state_dict']['conv2.weight']+ checkpoint_split_2['state_dict']['conv2.weight'] + checkpoint_split_3['state_dict']['conv2.weight'])/3
    #conv3_weight = (checkpoint_split_1['state_dict']['conv3.weight']+ checkpoint_split_2['state_dict']['conv3.weight'] + checkpoint_split_3['state_dict']['conv3.weight'])/3
    #conv4_weight = (checkpoint_split_1['state_dict']['conv4.weight']+ checkpoint_split_2['state_dict']['conv4.weight'] + checkpoint_split_3['state_dict']['conv4.weight'])/3

    #conv1_bias = (checkpoint_split_1['state_dict']['conv1.bias']+ checkpoint_split_2['state_dict']['conv1.bias'] + checkpoint_split_3['state_dict']['conv1.bias'])/3
    #conv2_bias = (checkpoint_split_1['state_dict']['conv2.bias']+ checkpoint_split_2['state_dict']['conv2.bias'] + checkpoint_split_3['state_dict']['conv2.bias'])/3
    #conv3_bias = (checkpoint_split_1['state_dict']['conv3.bias']+ checkpoint_split_2['state_dict']['conv3.bias'] + checkpoint_split_3['state_dict']['conv3.bias'])/3
    #conv4_bias = (checkpoint_split_1['state_dict']['conv4.bias']+ checkpoint_split_2['state_dict']['conv4.bias'] + checkpoint_split_3['state_dict']['conv4.bias'])/3
    
    net = net.to(cfg['devices'][0])
    
    net.conv1.weight = torch.nn.Parameter(checkpoint['state_dict']['conv1.weight'])
    net.conv2.weight = torch.nn.Parameter(checkpoint['state_dict']['conv2.weight'])
    net.conv3.weight = torch.nn.Parameter(checkpoint['state_dict']['conv3.weight'])
    net.conv4.weight = torch.nn.Parameter(checkpoint['state_dict']['conv4.weight'])

    net.conv1.bias = torch.nn.Parameter(checkpoint['state_dict']['conv1.bias'])
    net.conv2.bias = torch.nn.Parameter(checkpoint['state_dict']['conv2.bias'])
    net.conv3.bias = torch.nn.Parameter(checkpoint['state_dict']['conv3.bias'])
    net.conv4.bias = torch.nn.Parameter(checkpoint['state_dict']['conv4.bias'])

    #net.conv1.weight = torch.nn.Parameter(conv1_weight)
    #net.conv2.weight = torch.nn.Parameter(conv2_weight)
    #net.conv3.weight = torch.nn.Parameter(conv3_weight)
    #net.conv4.weight = torch.nn.Parameter(conv4_weight)

    #net.conv1.bias = torch.nn.Parameter(conv1_bias)
    #net.conv2.bias = torch.nn.Parameter(conv2_bias)
    #net.conv3.bias = torch.nn.Parameter(conv3_bias)
    #net.conv4.bias = torch.nn.Parameter(conv4_bias)


    extracted_features = {}
    for t_sbj in train_sbjs:
        extracted_features[t_sbj] = extarct_feature_one_epoch(train_loader[t_sbj], net, cfg['devices'][0])
        
    for v_sbj in val_sbjs: 
        extracted_features[v_sbj] = extarct_feature_one_epoch(test_loader[v_sbj], net, cfg['devices'][0])   
        
    #extracted_features = extarct_feature_one_epoch(train_loader, net, cfg['devices'][0])   
    del net
    return extracted_features


def extarct_feature_one_epoch(loader, network, gpu=None):
    features = []
    network.eval()
    with torch.no_grad():

        for i, (inputs, targets) in enumerate(loader):
            # send x and y to GPU
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)
                network = network.to(gpu)

            # send inputs through network to get predictions, loss and calculate softmax probabilities
            output = network(inputs)

            features.append(output)

    return features



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/60_frames_30_stride/deepconvlstm_long.yaml')
    parser.add_argument('--eval_type', default='split')
    parser.add_argument('--neptune', default=False, type=bool) 
    parser.add_argument('--seed', default=1, type=int)        
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)  

