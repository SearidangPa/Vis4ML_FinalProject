import pandas as pd

## Load the Dataset and the model Prediction
data_filename = '../Saved/Model/data.pkl'
df =  pd.read_pickle(data_filename)
X = df.drop(['target'], axis=1)
y = df['target']

# ----------------load all the attr values----------------
# shap 
shap_filename = '../Saved/Attr/Shap_nn.pkl'
shap_values_df = pd.read_pickle(shap_filename)

# Lime 
lime_weights_filename = '../Saved/Attr/lime_weights.pkl'
lime_weights_df = pd.read_pickle(lime_weights_filename)

# Integrated Gradients
ig_attr_filename = '../Saved/Attr/ig_attr.pkl'
ig_attr_df = pd.read_pickle(ig_attr_filename)

# Deep Lift
deepLift_filename = '../Saved/Attr/deepLift_attr.pkl'
deepLift_df = pd.read_pickle(deepLift_filename)

# melt 
shap_values_melted = shap_values_df.melt(ignore_index=False)
lime_weights_melted = lime_weights_df.melt(ignore_index=False)
ig_attr_melted = ig_attr_df.melt(ignore_index=False)
deepLift_attr_melted = deepLift_df.melt(ignore_index=False)

# ----------------dict----------------
dict_name_to_attr_df = {
    'shap': shap_values_df, 
    'lime': lime_weights_df, 
    'ig': ig_attr_df, 
    'deepLift': deepLift_df,
}

dict_name_to_rank_filename = {
    'shap': '../Saved/Attr/shap_rank.pkl', 
    'lime': '../Saved/Attr/lime_rank.pkl', 
    'ig': '../Saved/Attr/ig_rank.pkl', 
    'deepLift': '../Saved/Attr/deepLift_rank.pkl'
}

dict_method_name_to_attr_name = {
    'shap' : 'shap_value',
    'lime': 'lime_weight', 
    'ig':  'ig_attr', 
    'deepLift':'deepLift_attr' 
}

dict_name_to_attr_rank_df= {}
for method_name, rank_file_name in dict_name_to_rank_filename.items():
    dict_name_to_attr_rank_df[method_name]= pd.read_pickle(rank_file_name)


# ------------mean abs feature attr value and rank----------------
mean_attr = pd.DataFrame(shap_values_df.abs().mean()).reset_index()
mean_attr.columns = ['feature_name', 'shap_value']
mean_attr['lime_weight'] = lime_weights_df.abs().mean().values
mean_attr['ig_attr'] = ig_attr_df.abs().mean().values
mean_attr['deepLift_attr'] = deepLift_df.abs().mean().values

# compute the rank 
for method_name, attr_name in dict_method_name_to_attr_name.items():
    mean_attr[method_name + '_rank'] = mean_attr[attr_name].rank()

# compute the sum of rank of all methods
mean_attr['sum_rank'] = mean_attr[['shap_rank', 'lime_rank', 'ig_rank', 'deepLift_rank']].abs().sum(axis=1)

# melt for visualization
mean_attr_melted = mean_attr.melt(ignore_index = False, id_vars=['feature_name'])
mean_attr_melted.columns = ['feature_name', 'method', 'feature_attr']

# save mean_attr dataframe
mean_attr_filename = '../Saved/4vis/mean_attr.pkl'
mean_attr_melted.to_pickle(mean_attr_filename)

# ------------Individual DataPoint Visualization----------------
# Put all feature attributions into a single dataframe
Indiv_melted_filename = '../Saved/4vis/Indiv_melted.pkl'
Indiv_filename = '../Saved/4vis/Indiv.pkl'

Indiv_df = X.melt(ignore_index=False)
Indiv_df.columns = ['feature_name', 'feature_value']

Indiv_df['shap_value'] = shap_values_melted['value']
Indiv_df['lime_weight'] = lime_weights_melted['value']
Indiv_df['ig_attr'] = ig_attr_melted['value']
Indiv_df['deepLift_attr'] = deepLift_attr_melted['value']

for method_name, rank_df in dict_name_to_attr_rank_df.items():
    Indiv_df[method_name + '_rank'] = rank_df.melt(ignore_index=False)['value']

Indiv_df['sum_rank'] = Indiv_df[['shap_rank', 'lime_rank', 'ig_rank', 'deepLift_rank']].abs().sum(axis=1)

# save the indiv_df
Indiv_df.to_pickle(Indiv_filename)

# melt and save 
Indiv_melted = Indiv_df.melt(ignore_index = False, id_vars=['feature_name'])
Indiv_melted.columns = ['feature_name', 'method', 'feature_attr']
Indiv_melted.to_pickle(Indiv_melted_filename)




