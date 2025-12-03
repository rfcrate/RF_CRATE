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
    parser.add_argument("--cuda_index", type=int, default=0, help="The index of the cuda device")
    parser.add_argument("--mode", type=int, default=0, help="0: train + test, 1: test only, 2: finetune + test, 3: check the pipeline")
    parser.add_argument("--pretrained_model", type=str, default=None, help="The file path of the pretrained model weights")
    args = parser.parse_args()
    
    if args.mode == 1:
        print("Testing only!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 2:
        print("Finetuning and Testing!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 3:
        print("Check the pipeline!")
        num_epochs = 1
    else:
        print("Training and Testing!")
        
    # loading the configuration file #######################################
    config_file_name = args.config_file + '.yaml'
    all_yaml_files = []
    all_yaml_file_paths = []
    for root, dirs, files in os.walk("Configurations"):
        for file in files:
            if file.endswith(".yaml"):
                all_yaml_files.append(file)
                all_yaml_file_paths.append(os.path.join(root, file))
    if config_file_name not in all_yaml_files:
        print("Configuration file name is: ",config_file_name)
        print("The configuration file is not found! Please check the file name!")
        exit
    else:
        # if there are multiple configuration files with the same name, print all the locations
        if all_yaml_files.count(config_file_name) > 1:
            print("The configuration file is: ",config_file_name)
            print("There are multiple configuration files with the same name!")
            for i in range(all_yaml_files.count(config_file_name)):
                print(all_yaml_file_paths[all_yaml_files.index(config_file_name, i)])
            exit
        else:
            config_file_path = all_yaml_file_paths[all_yaml_files.index(config_file_name)]
            print("The configuration file is found at: ",config_file_path)
            with open(config_file_path, 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
    # traverse the configuration file if any value is a string: 'None', convert it to None
    for key, value in config.items():
        if value == 'None':
            config[key] = None
    config['cuda_index'] = args.cuda_index
    
    if args.mode == 3:
        config['num_epochs'] = 1
    
    config['tqdm_disable'] = False  #!!! set it to True if you don't want to see the tqdm bar
    
    tensorboard_folder = config['tensorboard_folder']
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    trained_model_folder = config['trained_model_folder']
    if not os.path.exists(trained_model_folder):
        os.makedirs(trained_model_folder)
    log_folder = config['log_folder']   # save all the outputs records when testing
    log_enable = config['log_enable']   # if log_enable is False, the log folder will not be created
    if not os.path.exists(log_folder) and log_enable:
        os.makedirs(log_folder)
    
    try: 
        model_save_enable = config['model_save_enable'] # if model_save_enable is False, the model weights will not be saved
    except:
        model_save_enable = True # by default, save the model weights
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda_index'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    
    # fix the seed for reproducibility
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    torch.cuda.manual_seed(config['init_rand_seed'])
    np.random.seed(config['init_rand_seed'])
    
    localtime = time.localtime(time.time())
    if args.mode == 1:
        log_file_name = f"{args.config_file}_{config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_test"
    elif args.mode == 2:
        log_file_name = f"{args.config_file}_{config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_finetune"
    else:
        log_file_name = f"{args.config_file}_{config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}"
    print("The log filename is: ",log_file_name)

    writer = SummaryWriter(config['tensorboard_folder'] + log_file_name)
    
    # configur the dataset and dataloader  ########################################
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
    elif config['dataset_name'] == 'OctoNetMini':
        data_sahpe_coverter = OctonetMini_data_shape_converter(config)
        dataset_get = OctonetMini
        dataloader_make = make_OctonetMini_dataloader
    elif config['dataset_name'] == 'RPI':
        data_sahpe_coverter = RPI_data_shape_converter(config)
        dataset_get = RPI_Dataset
        dataloader_make = make_RPI_dataloader
    else:
        print("The dataset name is wrong!")
        
    cross_domain = False
    if len(config['data_split']) ==3:
        print("The normal indomain experiments!")
        train_ratio = config['data_split'][0]
        val_ratio = config['data_split'][1]
        test_ratio = config['data_split'][2]
        merged_config = {**config, **config['all_dataset']} 
        dataset = dataset_get(merged_config)
        train_num = int(len(dataset)*train_ratio)
        val_num = int(len(dataset)*val_ratio)
        test_num = int(len(dataset)) - train_num - val_num
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num], generator=rng_generator)
    else:  # used for cross domain experiments
        print("The cross domain experiments!")
        train_set_config = {**config, **config['train_dataset']} 
        dataset = dataset_get(train_set_config)
        if config['dataset_name'] == 'widar3':
            # we need to lable_mapping for the cross domain experiments
            label_mapping = dataset.get_label_mapping()
            print("The label mapping is: ",label_mapping)
            config['label_mapping'] = label_mapping  # save the label mapping for the cross domain experiments
        val_num = int(len(dataset)*(0.2))
        indomain_test_num = int(len(dataset)*(0.2))  # this is used for the in-domain test
        train_num = len(dataset) - val_num - indomain_test_num
        train_set, val_set, indomain_test_set = torch.utils.data.random_split(dataset, [train_num, val_num, indomain_test_num], generator=rng_generator)
        print("finish the train, val, and indomain test loading")
        cross_domain = True
    
    train_loader = dataloader_make(train_set, is_training=True, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])
    validation_test_batchsize = min(config['batch_size'], 16)
    val_loader = dataloader_make(val_set, is_training=False, generator=rng_generator, batch_size=validation_test_batchsize,collate_fn_padd=None, num_workers=config['num_workers'])

    # configure the model, optimizer, scheduler, criterion, and metric  ########################################
    model_name = config['model_name']
    if 'rf_crate_recon' in model_name:
        model, decoder = get_registered_models(model_name, config)  # for the reconstruction model, we need to return the rf_crate model and the decoder
        model.to(device)
        decoder.to(device)  
    else:
        model = get_registered_models(model_name, config)
        decoder = None
        model.to(device)
    
    if args.mode == 1 or args.mode == 2:
        if args.mode == 2:
            state_dict = torch.load(args.pretrained_model)
            for key in list(state_dict.keys()):
                if 'mlp_head' in key:
                    del state_dict[key]
                    print("deleted:", key)
                if 'to_patch_embedding' in key:
                    del state_dict[key]
                    print("deleted:", key)
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_model))
        print("The pretrained model weights are loaded!")
    else:
        if config['resume_training']: # if resume training, the model weights filename is config['trained_model_folder']
            model.load_state_dict(torch.load(config['trained_model_folder'] + config['resume_training']))
            print("The model weights are loaded for resuming training!")
        
    if config['optimizer'] == "AdamW":
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=config['weight_decay'])                      
    elif config['optimizer'] == "Lion": # https://github.com/lucidrains/lion-pytorch
        optimizer = Lion(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == "Adam":
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=config['weight_decay'])
    else:
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=momentum, weight_decay=config['weight_decay'])
    try:
        warmup_steps = config['warmup_steps'] 
    except:
        warmup_steps = 10
    
    try:
        lower_lr_limit = config['lower_lr_limit']   
        lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), lower_lr_limit + (1 - lower_lr_limit) * (math.cos(step / config['num_epochs'] * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    except:
        lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), 0.5 * (math.cos(step / config['num_epochs'] * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if config['criterion'] == 'mse':
        criterion = nn.MSELoss()
    elif config['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['criterion'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    elif config['criterion'] == 'bone_loss':  # used for the 3D pose estimation: bone length loss + position loss (mse)
        criterion = BoneLength_Position_Loss()
    else:
        raise NotImplementedError
    criterion.to(device)
    
    # this is the subspace regularizationi setting for the rf_crate model
    try:
        subspace_regularization = config['ssr']
        regularizer_lambda = config['ssr_lambda']
        try:
            normalize_feature_magnitude = config['normalize_feature_magnitude']
        except:
            normalize_feature_magnitude = True
        if subspace_regularization == True:
            print("The subspace regularization is used")
            if config['model_name'] == 'rf_crate_tiny':
                regularizer = Subspace_Regularization(num_subspace = 6, dim = 384, normalize_feature_magnitude = normalize_feature_magnitude)
            elif config['model_name'] == 'rf_crate_small':
                regularizer = Subspace_Regularization(num_subspace = 12, dim = 576, normalize_feature_magnitude = normalize_feature_magnitude)
            elif config['model_name'] == 'rf_crate_base':
                regularizer = Subspace_Regularization(num_subspace = 12, dim = 768, normalize_feature_magnitude = normalize_feature_magnitude)
            elif config['model_name'] == 'rf_crate_large':
                regularizer = Subspace_Regularization(num_subspace = 16, dim = 1024, normalize_feature_magnitude = normalize_feature_magnitude)
            else:
                raise NotImplementedError
            regularizer.to(device)
        else:
            regularizer = None
            regularizer_lambda = 0
    except:
        regularizer = None
        regularizer_lambda = 0
    
    # the rotary physics prior embedding for rf_crate model
    try:
        if config['ppe'] is not None:
            print("The rotary physics prior embedding is used")
            rotary_physcis_prior_embedding = RotaryPhysicPriorEmbedding(config, device)
            # rotary_physcis_prior_embedding = MemoryOptimizedRotaryPhysicPriorEmbedding(config, device)
            rotary_physcis_prior_embedding.to(device)
        else:
            rotary_physcis_prior_embedding = None
    except:
        rotary_physcis_prior_embedding = None

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
    elif config['metric'] == 'mae':
        metric = MAE
    else:
        raise NotImplementedError
    
    # training/finetuning  ########################################
    if args.mode == 0:
        print("Start training!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,data_sahpe_coverter,criterion,regularizer,regularizer_lambda,
                                                     rotary_physcis_prior_embedding,optimizer,scheduler,metric,config,log_file_name,writer,device,decoder=decoder)
        print("Training is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    elif args.mode == 1:
        print("Start testing!")
    elif args.mode == 2 or args.mode == 3:
        print("Start finetuning or checking!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,data_sahpe_coverter,criterion,regularizer,regularizer_lambda,
                                                     rotary_physcis_prior_embedding,optimizer,scheduler,metric,config,log_file_name,writer,device,decoder=decoder)
        print("Finetuning or checking is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    else:
        print("The mode is wrong!")
    
    # testing ########################################
    model.eval()
    if cross_domain == True:
        # perform the in-domain test
        indomain_test_loader = dataloader_make(indomain_test_set, is_training=False, generator=rng_generator, batch_size=validation_test_batchsize,collate_fn_padd=None, num_workers=config['num_workers'])
        indomain_recordings, indomain_loss_all, indomain_recon_loss_all = inference(model,indomain_test_loader,data_sahpe_coverter,rotary_physcis_prior_embedding,config,device, criterion, metric, decoder=decoder)
        print(f'Average Loss (in-domain test set) {indomain_loss_all/ len(indomain_test_loader):.10f}, Average Reconstruction Loss (in-domain test set) {indomain_recon_loss_all/ len(indomain_test_loader):.10f}')
        indomain_metric_all = indomain_recordings['metrics']
        if len(indomain_metric_all) > 0:
            indomain_metric_all = np.mean(indomain_metric_all)
            writer.add_scalar('in-domain test metric (per sample)', indomain_metric_all, best_model_epoch)
        writer.add_scalar('in-domain test loss (per sample)', indomain_loss_all/ len(indomain_test_loader), best_model_epoch)
        if log_enable:
            indomain_test_result_save_path = config['log_folder']+ log_file_name + '_indomain_test.pkl'
            pickle.dump(indomain_recordings, open(indomain_test_result_save_path, 'wb'))
            print("The in-domain test results are saved at: ",indomain_test_result_save_path)
        
        del train_set, val_set, indomain_test_set
        test_set_config = {**config, **config['test_dataset']} 
        try:
            config['augmentation_ratio'] = 0
        except:
            pass
        test_set = dataset_get(test_set_config)
        print("finish the test loading")
    
    test_loader = dataloader_make(test_set, is_training=False, generator=rng_generator, batch_size=validation_test_batchsize,collate_fn_padd=None, num_workers=config['num_workers'])
    recordings, loss_all, recon_loss_all = inference(model,test_loader,data_sahpe_coverter,rotary_physcis_prior_embedding,config,device, criterion, metric, decoder=decoder)
    print(f'Average Loss (test set) {loss_all/ len(test_loader):.10f}, Average Reconstruction Loss (test set) {recon_loss_all/ len(test_loader):.10f}')
    metric_all = recordings['metrics']
    if len(metric_all) > 0:
        metric_all = np.mean(metric_all)
        writer.add_scalar('test metric (per sample)', metric_all, best_model_epoch)
    writer.add_scalar('test loss (per sample)', loss_all/ len(test_loader), best_model_epoch)
    if config['model_name'] == 'rf_crate_recon':
        writer.add_scalar('test recon loss (per sample)', recon_loss_all/ len(test_loader), best_model_epoch)
    writer.close()
    if log_enable:
        test_result_save_path = config['log_folder']+ log_file_name + '.pkl'
        pickle.dump(recordings, open(test_result_save_path, 'wb'))
        print("The test results are saved at: ",test_result_save_path)
