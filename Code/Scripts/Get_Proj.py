from sklearn.manifold import TSNE
import pandas as pd


def Get_tsne(df, dict_name_to_attr_df):
    tsne_filename = '../Saved/Proj/tsne.pkl'
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)

    # put the embedding of shap values into a df
    tsne_df = pd.DataFrame()
    tsne_df['target'] = df['target']

    for method_name, attr_df in dict_name_to_attr_df.items():
        attr_embedded = tsne.fit_transform(attr_df)
        tsne_df[method_name + '_x-tsne'] = attr_embedded[:, 0]
        tsne_df[method_name +'_y-tsne'] = attr_embedded[:, 1]
    
    # tsne of the dataset 
    dataset_embedded = tsne.fit_transform(df)
    tsne_df['data_x-tsne'] = dataset_embedded[:, 0]
    tsne_df['data_y-tsne'] = dataset_embedded[:, 1]

    # save the embedded attr dataframe
    tsne_df.to_pickle(tsne_filename)

def Get_tsne_feature_rank(df):
    tsne_rank_filename = '../Saved/Proj/tsne_rank.pkl'
    dict_name_to_rank_filename = {
        'shap': '../Saved/Attr/shap_rank.pkl', 
        'lime': '../Saved/Attr/lime_rank.pkl', 
        'ig': '../Saved/Attr/ig_rank.pkl', 
        'deepLift': '../Saved/Attr/deepLift_rank.pkl'
    }
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
        
    # compute t-sne proj for each signed rank feature attr method 
    tsne_feature_rank = pd.DataFrame()
    for method_name, rank_df in dict_name_to_rank_filename.items():
        rank_df = pd.read_pickle(dict_name_to_rank_filename[method_name])
        embedded = tsne.fit_transform(rank_df)
        tsne_feature_rank[method_name +'_rank_x-tsne'] = embedded[:, 0]
        tsne_feature_rank[method_name + '_rank_y-tsne'] = embedded[:, 1]
    
    tsne_feature_rank['target'] = df['target']
    tsne_feature_rank.to_pickle(tsne_rank_filename)


def main():
    # load the data and the attribution values
    data_filename = '../Saved/Model/data.pkl'
    df = pd.read_pickle(data_filename)    
    X = df.drop(['target'], axis=1)

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
    deepLift_attr_df = pd.read_pickle(deepLift_filename)

    dict_name_to_attr_df = {
        'shap': shap_values_df, 
        'lime': lime_weights_df, 
        'ig': ig_attr_df, 
        'deepLift': deepLift_attr_df,
    }

    # Generate the T-sne Projection
    Get_tsne(df, dict_name_to_attr_df)
    Get_tsne_feature_rank(df)





if __name__ == '__main__':
    main()