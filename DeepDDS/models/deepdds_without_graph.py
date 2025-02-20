import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Model without graph (using feature vectors)
class OHENet(torch.nn.Module):
    def __init__(self, num_features_drug=78, n_output=2, num_features_xt=954,  output_dim=128, dropout=0.2, file=None):
        super(OHENet, self).__init__()

        # Cell feature layers
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  output_dim * 2),
            nn.ReLU()
        )

        # Combined layers
        #self.fc1 = nn.Linear(num_features_drug * 2 + output_dim * 2, 1024) # For using cell line features
        self.fc1 = nn.Linear(num_features_drug * 2 + num_features_xt, 1024) # For using cell line one hot encoded features
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, drug1, drug2, cell):

        # Normalizing cell feature before concatenation
        #cell = F.normalize(cell, p=2, dim=1) # For using cell line features

        # Process cell feature vector
        #cell_vector = self.reduction(cell)   # For using cell line features

        
        # Concatenate drug1, drug2, and cell vectors
        #xc = torch.cat((drug1, drug2, cell_vector), 1)  # For using cell line features
        #xc = F.normalize(xc, p=2, dim=1) # For using cell line features

        xc = torch.cat((drug1, drug2, cell), 1)  # For using cell line one hot encoded features
        

        # Add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out