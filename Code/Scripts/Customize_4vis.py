import pandas as pd

## Load the Dataset and the model Prediction
data_filename = '../Saved/Model/data.pkl'
df =  pd.read_pickle(data_filename)
X = df.drop(['target'], axis=1)
y = df['target']

# ----------------load all the attr values----------------
shap_filename = '../Saved/Attr/Shap_nn.pkl'
lime_weights_filename = '../Saved/Attr/lime_weights.pkl'
ig_attr_filename = '../Saved/Attr/ig_attr.pkl'
deepLift_filename = '../Saved/Attr/deepLift_attr.pkl'

# ----------------dict----------------
name_to_attr_df = {
    'shap': pd.read_pickle(shap_filename), 
    'lime': pd.read_pickle(lime_weights_filename), 
    'ig': pd.read_pickle(ig_attr_filename), 
    'deepLift': pd.read_pickle(deepLift_filename),
}

name_to_rank_filename = {
    'shap': '../Saved/Attr/shap_rank.pkl', 
    'lime': '../Saved/Attr/lime_rank.pkl', 
    'ig': '../Saved/Attr/ig_rank.pkl', 
    'deepLift': '../Saved/Attr/deepLift_rank.pkl'
}

method_name_to_attr_name = {
    'shap' : 'shap_value',
    'lime': 'lime_weight', 
    'ig':  'ig_attr', 
    'deepLift':'deepLift_attr' 
}

name2rank_df= {}
for method_name, rank_file_name in name_to_rank_filename.items():
    name2rank_df[method_name]= pd.read_pickle(rank_file_name)

# ------------mean abs feature attr value and rank----------------
mean_attr = pd.DataFrame(data = X.columns, columns=['feature_name'])
for method_name, attr_df in name_to_attr_df.items():
    mean_attr[method_name_to_attr_name[method_name]] = attr_df.abs().mean().values

# compute the rank 
for method_name, attr_name in method_name_to_attr_name.items():
    mean_attr[method_name + '_rank'] = mean_attr[attr_name].rank()

# compute the sum of abs rank of all methods
mean_attr['sum_rank'] = mean_attr[['shap_rank', 'lime_rank', 'ig_rank', 'deepLift_rank']].abs().sum(axis=1)
# melt and save 
mean_attr_melted = mean_attr.melt(ignore_index = False, id_vars=['feature_name'])
mean_attr_melted.columns = ['feature_name', 'method', 'feature_attr']
mean_attr_filename = '../Saved/4vis/mean_attr.pkl'
mean_attr_melted.to_pickle(mean_attr_filename)

# # ------------Individual DataPoint Visualization----------------
# # Melt all feature attributions into a single dataframe
Indiv_melted_filename = '../Saved/4vis/Indiv_melted.pkl'
Indiv_filename = '../Saved/4vis/Indiv.pkl'
Indiv_df = X.melt(ignore_index=False)
Indiv_df.columns = ['feature_name', 'feature_value']

for method_name, attr_df in name_to_attr_df.items():
    Indiv_df[method_name_to_attr_name[method_name]] = attr_df.melt(ignore_index=False)['value']
    Indiv_df[method_name + '_rank'] = name2rank_df[method_name].melt(ignore_index=False)['value'] 

Indiv_df['sum_rank'] = Indiv_df[['shap_rank', 'lime_rank', 'ig_rank', 'deepLift_rank']].abs().sum(axis=1)

# save the indiv_df
Indiv_df.to_pickle(Indiv_filename)
""" 
    feature_name  feature_value  shap_value  ...  ig_rank  deepLift_rank  sum_rank
0       Pregnant            4.0   -0.005457  ...     -2.0           -2.0       9.0
1       Pregnant            4.0   -0.073003  ...      2.0           -1.0       7.0
..           ...            ...         ...  ...      ...            ...       ...
766          Age           67.0   -8.268724  ...     -7.0           -6.0      23.0
767          Age           28.0    0.104415  ...     -5.0           -6.0      18.0
"""

# melt and save Indiv_melted
Indiv_melted = Indiv_df.melt(ignore_index = False, id_vars=['feature_name'])
Indiv_melted.columns = ['feature_name', 'method', 'feature_attr']
Indiv_melted.to_pickle(Indiv_melted_filename)
""" 
    feature_name         method  feature_attr
0       Pregnant  feature_value           4.0
1       Pregnant  feature_value           4.0
..           ...            ...           ...
766          Age       sum_rank          23.0
767          Age       sum_rank          18.0
"""

##-----dataframe for subset analysis: feature_value, attr, abs_attr, signed_rank-----------
subset_analysis_df = df.copy(deep=True)

# rename columns to specify the type of values: 
# [feature_value, feature_attr, abs_feature_attr, signed_feature_rank]
for method_name, attr_df in name_to_attr_df.items():
    name2attrName = {}
    name2absAttrName = {}
    name2signedRankName = {}
    
    for feature_name in attr_df.columns:
        feature_attr_name = feature_name + '_' + method_name_to_attr_name[method_name]
        abs_feature_attr_name = 'abs_' + feature_attr_name
        attr_df[abs_feature_attr_name] = attr_df[feature_name].abs()

        name2attrName[feature_name] = feature_attr_name
        name2absAttrName[feature_name] = abs_feature_attr_name
        name2signedRankName[feature_name] = feature_name + '_' + method_name + '_rank'

    
    
    attr_df = attr_df.rename(columns= name2attrName)
    name2rank_df[method_name] = name2rank_df[method_name].rename(columns= name2signedRankName)

    subset_analysis_df = pd.concat([subset_analysis_df, attr_df], axis = 1)
    subset_analysis_df = pd.concat([subset_analysis_df, name2rank_df[method_name]], axis = 1)

# save the dataframe 
subset_analysis_filename = '../Saved/4vis/subset_analysis.pkl'
subset_analysis_df.to_pickle(subset_analysis_filename)



