import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import os

import config
from model import Net_View2, Net
from utils_data import split, load_data
from utils_model import train, valid, save_model, test, metric

import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(32)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    
    
def run(fold, repeat):
    setup_seed(42)

    transform = transforms.ToTensor()
    path_train = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'_train.csv'
    path_val = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'_val.csv'
    train_data = load_data(path_train, transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = load_data(path_val, transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    
    log_dir = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'.txt'
    f = open(log_dir,'w')
    f.close()

    model_View2 = Net_View2(model_config, drug_fp, cell)
    model = Net(model_config, model_View2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False)
    start_epoch = 0
    best_pear = 0
    save_dir = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'_best.pth'
    
    for epoch in range(start_epoch+1, epochs+1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        valid_loss, mse, rmse, r2, pear = valid(model, device, valid_loader, criterion)
        best_pear = save_model(pear, best_pear, epoch, model, optimizer, save_dir)
        if epoch%20 == 0:
            res = 'Epoch: {:05d}, Train loss:{:.3f}, Valid loss: {:.3f}, '.format(epoch, train_loss, valid_loss)
            res = res + 'MSE: {:.3f}, RMSE: {:.3f}, '.format(mse, rmse)
            res = res + 'R2: {:.3f}, Pearson: {:.3f}'.format(r2, pear)
            print(res)
            with open(log_dir,"a") as f:
                f.write(res+'\n')
                f.close()

                
def get_res(repeat):
    setup_seed(42)
    for fold in range(1,6):
        print(f'Start test, repeat {repeat}, fold {fold}')
        transform = transforms.ToTensor()
        path_test = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'_test.csv'
        test_data = load_data(path_test, transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model_View2 = Net_View2(model_config, drug_fp, cell)
        model = Net(model_config, model_View2)
        model.to(device)

        save_dir = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_fold'+str(fold)+'_best.pth'
        if os.path.exists(save_dir):
            checkpoint = torch.load(save_dir, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print('Load saved model')
        else:
            print('No saved model')
            break
        
        if fold == 1:
            test_data_pd = pd.read_csv(path_test)
            compare = test(model, device, test_loader)
            compare = compare.reset_index(drop=True)
            test_data_pd['true'] = compare['true']
            test_data_pd['pred'] = compare['pred']
        else:
            test_data_pd_temp = pd.read_csv(path_test)
            compare_temp = test(model, device, test_loader)
            compare_temp = compare_temp.reset_index(drop=True)
            test_data_pd_temp['true'] = compare_temp['true']
            test_data_pd_temp['pred'] = compare_temp['pred']
            test_data_pd = pd.concat([test_data_pd,test_data_pd_temp])
    
    test_data_pd = test_data_pd.drop(labels=['Unnamed: 0'],axis=1)
    
    pred_dir = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_predict.csv'
    test_data_pd.to_csv(pred_dir)
    
    mse, rmse, r2, pear = metric(test_data_pd)
    res = 'MSE: {:.3f}, RMSE: {:.3f}, '.format(mse, rmse)
    res = res + 'R2: {:.3f}, Pearson: {:.3f}'.format(r2, pear)
    print(res)
    
    res_dir = '/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/repeat'+str(repeat)+'_metric.txt'
    with open(res_dir, "w") as f:
        f.write(res+'\n')
        f.close()
        
        
if __name__ == '__main__':
    split_flag = 1 # split data
    train_flag = 1 # train model and save best model
    test_flag = 1 # test model and get predict
    
    if split_flag == 1:
        print('Split data')
        data = pd.read_csv('/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/data_to_split.csv')
        data = data.drop(columns='Unnamed: 0')
        print(data)

        for repeat in range(1,11):
            split(data, repeat=repeat)
            
    print('Load config')
    model_config = config.model_config
    gpu = model_config['gpu']
    batch_size = model_config['batch_size']
    criterion = nn.MSELoss()
    lr = model_config['lr']
    epochs = model_config['epochs']
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')
    
    #pair_graph = np.load('./rawData/Pair_graph.npy', allow_pickle=True)
    cell = pd.read_csv('/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/ohe_cell.csv')
    cell = cell.drop(labels=['id'],axis=1)
    drug_fp = pd.read_csv('/cta/users/ebcandir/JointSyn/Model/JointSyn_reg/rawData/ohe_drug.csv')
    drug_fp = drug_fp.drop(labels=['id'],axis=1)

    for repeat in range(1,11):
        for fold in range(1,6):
            if train_flag == 1:
                print(f'Start train, repeat {repeat}, fold {fold}')
                run(fold, repeat)
        if test_flag == 1:
            get_res(repeat)
        torch.cuda.empty_cache()