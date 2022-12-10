from captum.attr import IntegratedGradients, KernelShap, NoiseTunnel, DeepLift
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch 
import pandas as pd
from lime import lime_tabular
from sklearn.manifold import TSNE
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
 
    # Can handle numpy and tensor. Cannot handle panda dataframe
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            prob = torch.sigmoid(self.forward(x)).detach().numpy()
        return prob


# ----------------- SHAP ------------------
def Get_Shap_Attr(model, X):
    shap_filename = '../Saved/Attr/Shap_nn.pkl'
    ks = KernelShap(model)

    X_np = X.values.astype(np.float32)
    X_tensor = torch.tensor(X_np)

    attr = ks.attribute(X_tensor, n_samples=500)
    shap_values_df = pd.DataFrame(attr.numpy(), columns = X.columns)
    shap_values_df.to_pickle(shap_filename)
    return shap_values_df


#--------------------------LIME------------------------
# get the weights of each feature of the instance explanation 
def get_weights_from_exp(exp): 
    exp_list = exp.as_map()[0]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]
    return exp_weight

def Get_Lime_attr(model, X):
    lime_weights_filename = '../Saved/Attr/lime_weights.pkl'
    lime_infos_filename = '../Saved/Attr/lime_infos.pkl'
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

    # Create and save DataFrame
    lime_weights_df = pd.DataFrame(data=weights, columns=X.columns)
    lime_infos_df = pd.DataFrame(data=lime_infos, columns=['intercept','local_pred', 'model_pred'])
    lime_weights_df.to_pickle(lime_weights_filename)
    lime_infos_df.to_pickle(lime_infos_filename) 
    return lime_weights_df


#  -----------Integrated Gradient---------------

def Get_IG_attr(model, X):
    ig_attr_filename = '../Saved/Attr/ig_attr.pkl'
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
    return ig_attr_df

#  -----------DeepLift---------------

def Get_DeepLift_attr(model, X):
    deepLift_filename = '../Saved/Attr/deepLift_attr.pkl'
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
    
    # save into a dataframe
    deepLift_df = pd.DataFrame(data = attr, columns = X.columns)
    deepLift_df.to_pickle(deepLift_filename)
    return deepLift_df

#  ------------------------------------------

def ProcessData():
    data_filename = '../Saved/Model/data.pkl'
    df = pd.read_pickle(data_filename)
    
    # split
    X = df.drop(['target'], axis=1)
    y = df['target']

    X_normalized=(X - X.mean()) / X.std()
    return X_normalized, y

def Get_Attr_Signed_Rank(dict_name_to_attr_df, dict_name_to_rank_df):
    dict_name_to_rank_filename = {
        'shap': '../Saved/Attr/shap_rank.pkl', 
        'lime': '../Saved/Attr/lime_rank.pkl', 
        'ig': '../Saved/Attr/ig_rank.pkl', 
        'deepLift': '../Saved/Attr/deepLift_rank.pkl'
    }

    for method_name, rank_df in dict_name_to_rank_df.items():
        # get the rank
        for j in range (dict_name_to_attr_df[method_name].shape[0]):
            rank_df.loc[j] = dict_name_to_attr_df[method_name].loc[j].abs().rank()
    
        # get the sign 
        for feature in dict_name_to_attr_df[method_name].columns:
            rank_df.loc[dict_name_to_attr_df[method_name][feature] < 0, feature] *= -1
    
        # save the signed feature rank dataframe 
        rank_df.to_pickle(dict_name_to_rank_filename[method_name]) 

def main():
    # load the dataset and normalized the data
    X_normed, _ = ProcessData()

    # load model and data 
    model_path = '../Saved/Model/nn.pt'
    model = torch.load(model_path)

    # Generate feature attributions by various methods 
    shap_values_df = Get_Shap_Attr(model, X_normed)
    print("done computing shap_values")

    lime_weights_df = Get_Lime_attr(model, X_normed)
    print("done computing lime_weight")
    
    ig_attr_df = Get_IG_attr(model, X_normed)
    print("done computing ig_attr")
    
    deepLift_df = Get_DeepLift_attr(model, X_normed)
    print("done computing deepLift_attr")

    # Create empty dataframe for attr signed rank
    shap_rank = pd.DataFrame(pd.DataFrame(columns = X_normed.columns))
    lime_rank = pd.DataFrame(pd.DataFrame(columns = X_normed.columns))
    ig_rank = pd.DataFrame(pd.DataFrame(columns = X_normed.columns))
    deepLift_rank = pd.DataFrame(pd.DataFrame(columns = X_normed.columns)) 

    dict_name_to_attr_df = {
        'shap': shap_values_df, 
        'lime': lime_weights_df, 
        'ig': ig_attr_df, 
        'deepLift': deepLift_df,
    }
    dict_name_to_rank_df= {
        'shap': shap_rank, 
        'lime': lime_rank, 
        'ig': ig_rank, 
        'deepLift': deepLift_rank,
    }

    # Get the attr signed rank 
    Get_Attr_Signed_Rank(dict_name_to_attr_df, dict_name_to_rank_df)
   


if __name__ == '__main__':
    main()


