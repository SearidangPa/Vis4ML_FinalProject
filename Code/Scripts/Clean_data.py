import pmlb

def main():
    # filename to save to
    data_filename = '../Data/Model/data.pkl'

    # get the data from the internet
    df = pmlb.fetch_data('pima')

    # impute the missing input feature values with the median of the target class  
    imputeFeatures = ['plasma glucose', 'Diastolic blood pressure', 'Triceps skin fold thickness', 'Body mass index', '2-Hour serum insulin']
    for feature in imputeFeatures:
        df.loc[(df.target==0) & (df[feature] == 0), feature] = df[df.target==0][feature].median()
        df.loc[(df.target==1) & (df[feature] == 0), feature] = df[df.target==1][feature].median()
    

    # rename columns to have shorter names 
    new_col_dict = {
        "plasma glucose": "glucose",  
        "Diabetes pedigree function": "Pedigree", 
        "Diastolic blood pressure": "blood pressure", 
        "Triceps skin fold thickness": "skin thickness", 
        "Body mass index": "BMI", 
        "2-Hour serum insulin": "Insulin Level"
    }
    df = df.rename(columns = new_col_dict)


    # save the dataframe 
    df.to_pickle(data_filename)



if __name__ == '__main__':
    main()