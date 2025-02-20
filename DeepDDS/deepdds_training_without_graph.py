import csv
from itertools import islice
import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, confusion_matrix, cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from models.deepdds_without_graph import OHENet


# Training 
def train(model, device, loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(loader.dataset)))
    model.train()
    for batch_idx, (drug1_features, drug2_features, cell_features, y) in enumerate(loader):
        drug1_features, drug2_features, cell_features, y = drug1_features.to(device), drug2_features.to(device), cell_features.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(drug1_features, drug2_features, cell_features)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug1_features),
                                                                           len(loader.dataset),
                                                                           100. * batch_idx / len(loader),
                                                                           loss.item()))

# Prediction function
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for drug1_features, drug2_features, cell_features, y in loader:
            drug1_features, drug2_features, cell_features = drug1_features.to(device), drug2_features.to(device), cell_features.to(device)
            output = model(drug1_features, drug2_features, cell_features)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


modeling = OHENet

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'

# CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The code uses GPU...' if torch.cuda.is_available() else 'The code uses CPU!!!')

drug1_path = '/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_drug1_features_ohe.csv'
drug2_path = '/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_drug2_features_ohe.csv'
#cell_path = '/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_cell_features.csv' #  For using cell line features
cell_path = '/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_cell_features_ohe.csv'  # For using cell line one hot encoded features
label_path = '/cta/users/ebcandir/DeepDDs/data/new_labels_0_10.csv'

drug1_features = pd.read_csv(drug1_path).values  
drug2_features = pd.read_csv(drug2_path).values  
cell_features = pd.read_csv(cell_path).values    
labels = pd.read_csv(label_path)['label'].values 

drug1_features_tensor = torch.tensor(drug1_features, dtype=torch.float32)
drug2_features_tensor = torch.tensor(drug2_features, dtype=torch.float32)
cell_features_tensor = torch.tensor(cell_features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(drug1_features_tensor, drug2_features_tensor, cell_features_tensor, labels_tensor)

# Split dataset
lenth = len(dataset)
pot = int(lenth / 5)

#random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    #test_num = random_num[pot*i:pot*(i+1)]
    #train_num = random_num[:pot*i] + random_num[pot*(i+1):]
    
    train_ind = list(np.loadtxt("/cta/users/ebcandir/DeepDDs/data/train_indices_fold_" + str(i + 1) +".txt",dtype=int))
    dataset_train = torch.utils.data.Subset(dataset, train_ind)

    test_ind = list(np.loadtxt("/cta/users/ebcandir/DeepDDs/data/test_indices_fold_" + str(i + 1) +".txt",dtype=int))
    dataset_test = torch.utils.data.Subset(dataset, test_ind)


    loader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    loader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=None)

    model = modeling(num_features_drug=len(dataset[0][0]), num_features_xt=len(dataset[0][2])).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model_file_name = f'/cta/users/ebcandir/DeepDDs/data/result/OHENet(DrugA_DrugB){i}--model_{datafile}.model'
    file_AUCs = f'/cta/users/ebcandir/DeepDDs/data/result/OHENet(DrugA_DrugB){i}--AUCs--{datafile}.txt'
    AUCs_header = 'Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL'

    with open(file_AUCs, 'w') as f:
        f.write(AUCs_header + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, loader_test)

        # Compute performance metrics
        AUC = roc_auc_score(T, S)
        precision, recall, _ = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        RECALL = recall_score(T, Y)

        if best_auc < AUC:
            best_auc = AUC
            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL]
            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')
            torch.save(model.state_dict(), model_file_name)

        print('best_auc', best_auc)
