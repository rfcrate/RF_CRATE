import cv2
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import yaml
import os
import argparse
import torch
from tqdm import tqdm
import math
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import timm
from timm.scheduler import CosineLRScheduler
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from lion_pytorch import Lion
# Datasets
from Datasets import *
# models
from Models import *
from func_utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and Testing')
    parser.add_argument("--config_file", type=str, help="Configuration YAML file")
    args = parser.parse_args()

    with open('Configurations/' + args.config_file + '.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    torch.autograd.set_detect_anomaly(False)
    
    config['tqdm_disable'] = False
    
    tensorboard_folder = config['tensorboard_folder']
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    trained_model_folder = config['trained_model_folder']
    if not os.path.exists(trained_model_folder):
        os.makedirs(trained_model_folder)
    log_folder = config['log_folder']
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda_index'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    
    # fix the seed for reproducibility
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    torch.cuda.manual_seed(config['init_rand_seed'])
    np.random.seed(config['init_rand_seed'])
    
    localtime = time.localtime(time.time())
    # index_of_experiment = len(os.listdir(tensorboard_folder))
    # the name for the log file, saved weights name, and the tensorboard log
    # log_file_name = args.config_file + "_" + config['model_name'] + "_" + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec) 
    log_file_name = f"{args.config_file}_{config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}"
    print("The log file name is: ",log_file_name)

    writer = SummaryWriter(config['tensorboard_folder'] + log_file_name)
    
    # configur the dataset and dataloader
    if config['dataset_name'] == 'widar3':
        data_sahpe_coverter = widar3_data_shape_converter(config)
        dataset_get = Get_Widar3_Dataset
        dataloader_make = make_widar3_dataloader
    elif config['dataset_name'] == 'widar_gait':
        data_sahpe_coverter = widarGait_data_shape_converter(config)
        dataset_get = WidarGait_Dataset
        dataloader_make = make_widar_gait_dataloader
    elif config['dataset_name'] == 'HuPR':  
        data_sahpe_coverter = HuPR_data_shape_converter(config)
        dataset_get = HuPR_Dataset
        dataloader_make = make_HuPR_dataloader
    elif config['dataset_name'] == 'OPERAnet_UWB':
        data_sahpe_coverter = OPERAnet_UWB_data_shape_converter(config)
        dataset_get = OPERAnet_UWB_Dataset
        dataloader_make = make_OPERAnet_UWB_dataloader
    else:
        print("The dataset name is wrong!")
        
    cross_domain = False
    if len(config['data_split']) ==3:
        train_ratio = config['data_split'][0]
        val_ratio = config['data_split'][1]
        test_ratio = config['data_split'][2]
        merged_config = {**config, **config['all_dataset']} 
        dataset = dataset_get(merged_config)
        # split the dataset with the given ratio: train_ratio, val_ratio, test_ratio
        train_num = int(len(dataset)*train_ratio)
        val_num = int(len(dataset)*val_ratio)
        test_num = int(len(dataset)) - train_num - val_num
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num], generator=rng_generator)
    else:  # used for cross domain experiments
        train_set_config = {**config, **config['train_dataset']} 
        train_set = dataset_get(train_set_config)
        val_num = int(len(train_set)*(0.2))
        indomain_test_num = int(len(train_set)*(0.2))  # this is used for the in-domain test
        train_num = len(train_set) - val_num - indomain_test_num
        # train_set, val_set = torch.utils.data.random_split(train_set, [train_num, val_num], generator=rng_generator)
        train_set, val_set, indomain_test_set = torch.utils.data.random_split(train_set, [train_num, val_num, indomain_test_num], generator=rng_generator)
        cross_domain = True
    
    train_loader = dataloader_make(train_set, is_training=True, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])
    val_loader = dataloader_make(val_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])


    model_name = config['model_name']
    model = get_registered_models(model_name, config)
    model.to(device)
    if config['resume_training']:
        model.load_state_dict(torch.load(config['trained_model_folder'] + config['resume_training']))
    
    if config['optimizer'] == "AdamW":
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=config['weight_decay'])                      
    elif config['optimizer'] == "Lion": # https://github.com/lucidrains/lion-pytorch
        optimizer = Lion(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=momentum, weight_decay=config['weight_decay'])
    
    warmup_steps = 10
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), 0.5 * (math.cos(step / config['num_epochs'] * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if config['criterion'] == 'mse':
        criterion = nn.MSELoss()
    elif config['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['criterion'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        raise NotImplementedError
    criterion.to(device)

    # 'accuracy', 'f1_score', 'precision', 'recall', 'mpjpe_2d', 'mpjpe_3d'
    if config['metric'] == 'accuracy':
        metric = accuracy
    elif config['metric'] == 'f1_score':
        metric = f1_score
    elif config['metric'] == 'precision':
        metric = precision
    elif config['metric'] == 'recall':
        metric = recall
    elif config['metric'] == 'mpjpe_2d':
        metric = MPJPE_2D
    elif config['metric'] == 'mpjpe_3d':
        metric = MPJPE_3D
    else:
        raise NotImplementedError
    
    
    print("Start training!")
    saved_model_path, best_model_epoch = train(model,train_loader,val_loader,data_sahpe_coverter,criterion,optimizer,scheduler,metric,config,log_file_name,writer,device)
    print("Training is done!")
    print("The best model is at epoch: ",best_model_epoch)

    best_model_weights = torch.load(saved_model_path)
    model.load_state_dict(best_model_weights)
    model.eval()
    if cross_domain == True:
        # perform the in-domain test
        indomain_test_loader = dataloader_make(indomain_test_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])
        indomain_recordings, indomain_loss_all = inference(model,indomain_test_loader,data_sahpe_coverter,config,device, criterion, metric)
        print(f'Average Loss (in-domain test set) {indomain_loss_all/ len(indomain_test_loader):.10f}')
        indomain_metric_all = indomain_recordings['metrics']
        if len(indomain_metric_all) > 0:
            indomain_metric_all = np.mean(indomain_metric_all)
            writer.add_scalar('in-domain test metric (per sample)', indomain_metric_all, best_model_epoch)
        writer.add_scalar('in-domain test loss (per sample)', indomain_loss_all/ len(indomain_test_loader), best_model_epoch)
        indomain_test_result_save_path = config['log_folder']+ log_file_name + '_indomain_test.pkl'
        pickle.dump(indomain_recordings, open(indomain_test_result_save_path, 'wb'))
        print("The in-domain test results are saved at: ",indomain_test_result_save_path)
        # perform the cross-domain test
        test_set_config = {**config, **config['test_dataset']} 
        test_set = dataset_get(test_set_config)
    test_loader = dataloader_make(test_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])
    recordings, loss_all = inference(model,test_loader,data_sahpe_coverter,config,device, criterion, metric)
    print(f'Average Loss (test set) {loss_all/ len(test_loader):.10f}')
    metric_all = recordings['metrics']
    if len(metric_all) > 0:
        metric_all = np.mean(metric_all)
        writer.add_scalar('test metric (per sample)', metric_all, best_model_epoch)
    writer.add_scalar('test loss (per sample)', loss_all/ len(test_loader), best_model_epoch)
    writer.close()
    test_result_save_path = config['log_folder']+ log_file_name + '.pkl'
    # add the configuration to the recordings
    recordings['config'] = config
    pickle.dump(recordings, open(test_result_save_path, 'wb'))
    print("The test results are saved at: ",test_result_save_path)
