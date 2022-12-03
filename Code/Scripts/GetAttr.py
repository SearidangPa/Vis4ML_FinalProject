from captum.attr import IntegratedGradients, KernelShap, NoiseTunnel, DeepLift
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch 
import pmlb
import pandas as pd
from lime import lime_tabular
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
pd.options.display.float_format = '{:.3f}'.format

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
 
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            prob = torch.sigmoid(self.forward(x)).detach().numpy()
        return prob

def ProcessData(df):
    # impute the missing input feature values with the median of the target class  
    imputeFeatures = ['plasma glucose', 'Diastolic blood pressure', 'Triceps skin fold thickness', 'Body mass index', '2-Hour serum insulin']
    for feature in imputeFeatures:
        df.loc[(df.target==0) & (df[feature] == 0), feature] = df[df.target==0][feature].median()
        df.loc[(df.target==1) & (df[feature] == 0), feature] = df[df.target==1][feature].median()
    
    # split
    X = df.drop(['target'], axis=1)
    y = df['target']

    X_normalized=(X - X.mean()) / X.std()
    return X_normalized, y


# ----------------- SHAP ------------------
def Get_Shap_Attr(model, X):
    shap_filename = '../Weights/Attr/Shap_nn.pkl'
    ks = KernelShap(model)

    X_np = X.values.astype(np.float32)
    X_tensor = torch.tensor(X_np)

    attr = ks.attribute(X_tensor, n_samples=500)
    shap_df = pd.DataFrame(attr.numpy(), columns = X.columns)
    shap_df.to_pickle(shap_filename)


#--------------------------LIME------------------------
def Get_Lime_attr(model, X):
    lime_weights_filename = '../Weights/Attr/lime_weights.pkl'
    lime_infos_filename = '../Weights/Attr/lime_infos.pkl'
    X_np = X.values.astype(np.float32)
    feature_names = list(X.columns)

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data = X_np,
        feature_names = feature_names,
        mode = 'classification',
        kernel_width = 0.75,
        random_state = 42
    )

    weights = []
    lime_infos = []
    for i in range(X.values.shape[0]):
        # get explanation for instance i
        exp = lime_explainer.explain_instance(
            data_row= X_np[i],
            num_samples = 500,
            labels = [0],
            predict_fn = model.predict
        )
        # Get weights
        weight = get_weights_from_exp(exp)
        weights.append(weight)
        lime_data = [exp.intercept[0], exp.local_pred[0], model.predict(X_np[i])[0]]
        lime_infos.append(lime_data)

    # Create DataFrame
    lime_weights_df = pd.DataFrame(data=weights, columns=X.columns)
    lime_infos_df = pd.DataFrame(data=lime_infos, columns=['intercept','local_pred', 'model_pred'])
    lime_weights_df.to_pickle(lime_weights_filename)
    lime_infos_df.to_pickle(lime_infos_filename) 


# get the weights of each feature of the instance explanation 
def get_weights_from_exp(exp): 
    exp_list = exp.as_map()[0]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]
    return exp_weight

#  -----------Integrated Gradient---------------
# to-do: try a different baseline 
def Get_IG_attr(model, X):
    ig_attr_filename = '../Weights/Attr/ig_attr.pkl'
    ig = IntegratedGradients(model)
    X_tensor_grad = torch.tensor(X.values.astype(np.float32)).requires_grad_()   
    
    nt = NoiseTunnel(ig)

    attr, delta = nt.attribute(
        X_tensor_grad, 
        nt_type ='smoothgrad',
        nt_samples=10,
        return_convergence_delta=True
    )
    attr = attr.detach().numpy()
    
    # save into a dataframe
    ig_attr_df = pd.DataFrame(data = attr, columns = X.columns)
    ig_attr_df.to_pickle(ig_attr_filename)

def Get_DeepLift_attr(model, X):
    deepLift_filename = '../Weights/Attr/deepLift_attr.pkl'
    dl = DeepLift(model)
    X_tensor_grad = torch.tensor(X.values.astype(np.float32)).requires_grad_()   
    
    nt = NoiseTunnel(dl)

    attr, delta = nt.attribute(
        X_tensor_grad, 
        nt_type ='smoothgrad',
        nt_samples=10,
        return_convergence_delta=True
    )
    attr = attr.detach().numpy()
    print(delta.detach().numpy())
    
    # save into a dataframe
    deepLift_df = pd.DataFrame(data = attr, columns = X.columns)
    deepLift_df.to_pickle(deepLift_filename)


def main():
    model_path = '../Weights/Model/nn.pt'
    # load model and data 
    df = pmlb.fetch_data('pima')
    model = torch.load(model_path)
    X, y = ProcessData(df)
    
    # Generate feature attributions by various methods 
    # Get_Shap_Attr(model, X)
    # Get_Lime_attr(model, X)
    # Get_IG_attr(model, X)
    Get_DeepLift_attr(model, X)

    
   






if __name__ == '__main__':
    main()


