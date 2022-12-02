from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import torch
import torch.nn as nn
import pmlb
import pickle
from sklearn.model_selection import train_test_split
from lime import lime_tabular
import shap

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

def main():
    model_path = '../Weights/Model/nn.pt'
    shap_filename = 'Weights/Attr/ShapNN.pkl'

    model = torch.load(model_path)

    df = pmlb.fetch_data('pima')
    X_train_df, X_test_df, y_train_df, y_test_df = ProcessData(df)

    feature_names = list(X_test_df.columns)

    # SHAP

    # ig = IntegratedGradients(model)
    # X_test = torch.tensor(X_test_df.values.astype(np.float32)).requires_grad_()   
    # feature_attr, delta = ig.attribute(X_test, return_convergence_delta=True)
    # attr = feature_attr.detach().numpy()


if __name__ == '__main__':
    main()


