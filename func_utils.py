from tqdm import tqdm
import torch
import os
from Datasets import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# get the id of the available i-th gpu
def get_gpu_ids(i):
    gpu_ids = []
    gpu_info = os.popen("nvidia-smi -L").readlines()
    for line in gpu_info:
        # print(line)
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        gpu_ids.append(ids)
    if i >= len(gpu_ids):
        print("The number of the gpu is not enough! using the 0 by default")
        return gpu_ids[0]
    return gpu_ids[i]


# excute the model and get the predictions and the labels
def inference(model,data_loader,data_sahpe_coverter,config,device, criterion =None, metric = None):
    model.eval()  # Set the model to evaluation mode
    recordings = {
        'predicts': [],
        'attributes': [],
        'labels': [],
        'metrics': [],
    }
    
    loss_all = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, sample in enumerate(tqdm(data_loader, disable= config['tqdm_disable'])):
            data, label, *attr = sample
            recordings['attributes'].append(attr)
            if data_sahpe_coverter is not None:
                input = data_sahpe_coverter.shape_convert(data)
            else:
                input = data
            if config['format'] == 'complex':
                input = input.cfloat().to(device)
            elif config['format'] == 'dfs' and config['model_input_shape'] == 'BCHW-C':
                input = input.cfloat().to(device)
            else:
                input = input.float().to(device)   
            predicts = model(input)
            
            # if the label is more than two dimensions: (Batch_size, dim_num, ...) -> (Batch_size, ...)
            if len(label.shape) > 2:
                label = label.squeeze()
                label = label.reshape(label.shape[0], -1)
                
            if criterion is not None: # if the loss function is not None, then calculate the loss
                if config['criterion'] == 'mse':
                    label = label.float().to(device)
                else:
                    label = label.long().to(device) 
                loss = criterion(predicts, label)
                loss_all += loss.item()
            if metric is not None:
                metric_value = metric(predicts, label)
                recordings['metrics'].append(metric_value)
            recordings['labels'].append(label.cpu().numpy())
            recordings['predicts'].append(predicts.cpu().numpy())
    return recordings, loss_all


# model training
def train(model,train_loader,validation_loader,data_sahpe_coverter,criterion,optimizer,scheduler, metric,config,log_file_name,writer,device):
    best_loss = float('inf')
    best_model_weights = None
    best_model_epoch = 0

    for epoch in range(config['num_epochs']):
        model.train()  # Set the model to training mode
        loss_epoch = 0
        metric_epoch = 0
        for i, sample in enumerate(tqdm(train_loader ,disable= config['tqdm_disable'])):
            data, label, *attr = sample
            if data_sahpe_coverter is not None:
                input = data_sahpe_coverter.shape_convert(data)
            else:
                input = data
            loss = 0
            # with torch.autograd.detect_anomaly():
            if config['format'] == 'complex':
                input = input.cfloat().to(device)
            elif config['format'] == 'dfs' and config['model_input_shape'] == 'BCHW-C':
                input = input.cfloat().to(device)
            else:
                input = input.float().to(device)  
            predicts = model(input)
            
            if config['criterion'] == 'mse':
                label = label.float().to(device)
            else:
                label = label.long().to(device) 
            # if the label is more than two dimensions: (Batch_size, dim_num, ...) -> (Batch_size, ...)
            if len(label.shape) > 2:
                label = label.squeeze()
                label = label.reshape(label.shape[0], -1)
            
            loss = criterion(predicts, label)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            writer.add_scalar('train loss (per batch)', loss.item(), epoch * len(train_loader) + i)
            if metric is not None:
                metric_value = metric(predicts, label)
                metric_epoch += metric_value
                writer.add_scalar('train metric (per batch)', metric_value, epoch * len(train_loader) + i)
            
        if scheduler is not None:
            scheduler.step()
        print(f'Epoch {epoch}, Average Loss (train set) {loss_epoch/ len(train_loader):.10f}')
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train loss (per sample)', loss_epoch/ len(train_loader), epoch)
        if metric is not None:
            writer.add_scalar('train metric (per sample)', metric_epoch/ len(train_loader), epoch)
        
        recordings, loss_all = inference(model,validation_loader,data_sahpe_coverter,config,device, criterion, metric)
        metric_all = recordings['metrics']
        if len(metric_all) > 0:
            metric_all = np.mean(metric_all)
            writer.add_scalar('validation metric (per sample)', metric_all, epoch)
        writer.add_scalar('validation loss (per sample)', loss_all/ len(validation_loader), epoch)
        print(f'Epoch {epoch}, Average Loss (valid set) {loss_all/ len(validation_loader):.10f}')
        
        if loss_all<best_loss:
            best_model_weights = model.state_dict()
            best_loss = loss_all
            best_model_epoch = epoch
        else:
            if epoch - best_model_epoch >= config['early_stop']:    
                print('early stop with best model at epoch: ',best_model_epoch)
                break
        # if the loss is nan, then stop the training
        if np.isnan(loss_all):
            print('early nan stop with best model at epoch: ',best_model_epoch)
            break
    saved_model_path = config['trained_model_folder']+ log_file_name + '.pth'
    torch.save(best_model_weights, saved_model_path)
    return saved_model_path, best_model_epoch





# metrics calculation

def accuracy(predicts, labels):
    # predicts: (batch_size, num_classes)
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    return np.sum(predicts == labels) / len(labels)

def precision(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fp = np.sum((predicts != labels) & (predicts == 1))
    return tp / (tp + fp)

def recall(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fn = np.sum((predicts != labels) & (predicts == 0))
    return tp / (tp + fn)

def f1_score(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fp = np.sum((predicts != labels) & (predicts == 1))
    fn = np.sum((predicts != labels) & (predicts == 0))
    return tp / (tp + 0.5 * (fp + fn))


# mean per joint position error
def MPJPE_2D(predicts, labels, per_sample = False):
    # if the shape of the predicts and labels are : (batch_size, num_joints * 2)
    # reshape the predicts and labels to (batch_size, num_joints, dim)
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = predicts.reshape(predicts.shape[0], -1, 2)
    labels = labels.reshape(labels.shape[0], -1, 2)
    per_sample_mpjpe = np.mean(np.linalg.norm(predicts - labels, axis=2), axis=1)
    if per_sample:
        return per_sample_mpjpe
    return np.mean(per_sample_mpjpe, axis=0)

def MPJPE_3D(predicts, labels, per_sample = False):
    # if the shape of the predicts and labels are : (batch_size, num_joints * 3) 
    # reshape the predicts and labels to (batch_size, num_joints, dim)
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = predicts.reshape(predicts.shape[0], -1, 3)
    labels = labels.reshape(labels.shape[0], -1, 3)
    per_sample_mpjpe = np.mean(np.linalg.norm(predicts - labels, axis=2), axis=1)
    if per_sample:
        return per_sample_mpjpe
    return np.mean(per_sample_mpjpe, axis=0)