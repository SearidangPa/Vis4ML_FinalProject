import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


"""
I took a lot of inspiration from this blog post 
Reference: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
"""

class BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class PimaDataset(Dataset):
    def __init__(self, X_df, y_df):
        X = X_df.copy()
        y = y_df.copy()
        self.X = torch.tensor(X.values.astype(np.float32))
        self.y = torch.tensor(y.values.astype(np.float32))

    def __len__(self):  
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def Train_Model(model, X_train_df, y_train_df, class_ratio):
    pos_weight = class_ratio * torch.ones([1])
    criterion = nn.BCEWithLogitsLoss(pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay = 0.0001)

    train_Dataset = PimaDataset(X_train_df, y_train_df)
    train_loader = DataLoader(train_Dataset, batch_size=64)

    model.train()
    lambda1 = 0.00175

    for e in range(2000):
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
            acc = batch_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        if e % 100 == 0:    
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

def batch_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    num_correct_results = (y_pred_tag == y_test).sum().float()
    acc = num_correct_results/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def Eval_Model(model, X_test_df, y_test_df):
    test_Dataset = PimaDataset(X_test_df, y_test_df)
    test_loader = DataLoader(dataset = test_Dataset, batch_size = 1)

    y_pred_list = []
    model.eval()
    with torch.no_grad():
         for X_test, _ in test_loader:
            y_test_pred = torch.sigmoid(model(X_test))
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.numpy())
            
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print("test accuracy: ", accuracy_score(y_pred_list, y_test_df))

    # confusion matrix
    df_test = X_test_df.copy()
    df_test['prediction'] = y_pred_list
    df_test['target'] = y_test_df
    cm = df_test.groupby(['target', 'prediction'], as_index=False).size()
    print(cm)

def ProcessData():
    # load the processed dataframe 
    data_filename = '../data/Model/data.pkl'
    df = pd.read_pickle(data_filename)

    # split
    X = df.drop(['target'], axis=1)
    Y = df['target']

    # normalize each column of the data/    X_normalized=(X - X.mean()) / X.std()

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_normalized, Y, test_size=0.25, random_state=42)
    return X_train_df, X_test_df, y_train_df, y_test_df


def main():
    model_path = '../data/Model/new.pt'
    
    # load the data/    X_train_df, X_test_df, y_train_df, y_test_df = ProcessData()
    
    # train the model
    model = BinaryClassification()    
    class_ratio = sum(df['target'] == 0) / sum(df['target'] == 1)
    Train_Model(model, X_train_df, y_train_df, class_ratio)

    # Print out accuracy score and confusion matrix
    Eval_Model(model, X_test_df, y_test_df)

    #save the model
    torch.save(model, model_path)

   

if __name__ == '__main__':
    main()