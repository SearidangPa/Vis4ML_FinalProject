import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pmlb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import random

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt


"""
I took a lot of inspiration from this blog post 
Reference: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
"""

class PimaDataset(Dataset):
    def __init__(self, X_df, y_df, training_Mode):
        X = X_df.copy()
        y = y_df.copy()
        self.X = torch.tensor(X.values.astype(np.float32))
        self.y = torch.tensor(y.values.astype(np.float32))

    def __len__(self):  
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout2 = nn.Dropout(p=0.1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def Train_Model(model, X_train_df, y_train_df, X_test_df, y_test_df):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay = 0.0001)

    train_Dataset = PimaDataset(X_train_df, y_train_df, training_Mode = True)
    train_loader = DataLoader(train_Dataset, batch_size=64)

    lambda1 = 0.00175
    for e in range(2000):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            params_fc1 = torch.cat([x.view(-1) for x in model.fc1.parameters()])
            params_fc2 = torch.cat([x.view(-1) for x in model.fc2.parameters()])
            params_fc3 = torch.cat([x.view(-1) for x in model.fc3.parameters()])

            l1_regularization = 0
            for p in [params_fc1, params_fc2, params_fc3]:
                l1_regularization = l1_regularization + torch.norm(p, 1)

            loss = criterion(y_pred, y_batch.unsqueeze(1)) + lambda1 * l1_regularization
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        if e % 100 == 0:    
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            Eval_Model(model, X_test_df, y_test_df)

def Eval_Model(model, X_test_df, y_test_df):
    test_Dataset = PimaDataset(X_test_df, y_test_df, training_Mode = False)
    test_loader = DataLoader(dataset = test_Dataset, batch_size = 1)

    y_pred_list = []
    model.eval()
    with torch.no_grad():
         for X_test, _ in test_loader:
            y_test_pred = model(X_test)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.numpy())
            
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print("test accuracy: ", accuracy_score(y_pred_list, y_test_df))

    df_test = X_test_df.copy()
    df_test['prediction'] = y_pred_list
    df_test['target'] = y_test_df
    cm = df_test.groupby(['target', 'prediction'], as_index=False).size()
    print(cm)


    

def ProcessData(df):
    # impute the missing input feature values with the median of the target class  
    imputeFeatures = ['plasma glucose', 'Diastolic blood pressure', 'Triceps skin fold thickness', 'Body mass index', '2-Hour serum insulin']
    for feature in imputeFeatures:
        df.loc[(df.target==0) & (df[feature] == 0), feature] = df[df.target==0][feature].median()
        df.loc[(df.target==1) & (df[feature] == 0), feature] = df[df.target==1][feature].median()
    
    # split
    X = df.drop(['target'], axis=1)
    Y = df['target']
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, Y, test_size=0.25, random_state=42)
    return X_train_df, X_test_df, y_train_df, y_test_df

# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)

def main():
    model_path = './SavedWeights/nn2.pt'
    df = pmlb.fetch_data('pima')
    
    X_train_df, X_test_df, y_train_df, y_test_df = ProcessData(df)
    
    trained = True

    if trained == True:
        model = torch.load(model_path)
    else:
        model = BinaryClassification()    
        Train_Model(model, X_train_df, y_train_df, X_test_df, y_test_df)
        torch.save(model, model_path)

    Eval_Model(model, X_test_df, y_test_df)
    
    feature_names = list(X_test_df.columns)

    ig = IntegratedGradients(model)
    X_test = torch.tensor(X_test_df.values.astype(np.float32)).requires_grad_()   
    feature_attr, delta = ig.attribute(X_test, return_convergence_delta=True)
    attr = feature_attr.detach().numpy()

    visualize_importances(feature_names, np.mean(attr, axis=0))

   

if __name__ == '__main__':
    main()