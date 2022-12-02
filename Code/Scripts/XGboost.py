from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
import pandas as pd
import pmlb
import numpy as np
import os
import pickle
import xgboost 



model_filename = '../Weights/Model/XGboost.sav'

df = pmlb.fetch_data('pima')

# impute the missing input feature values with the median of the target class  
imputeFeatures = ['plasma glucose', 'Diastolic blood pressure', 'Triceps skin fold thickness', 'Body mass index', '2-Hour serum insulin']
for feature in imputeFeatures:
    df.loc[(df.target==0) & (df[feature] == 0), feature] = df[df.target==0][feature].median()
    df.loc[(df.target==1) & (df[feature] == 0), feature] = df[df.target==1][feature].median()

# split
X = df.drop(['target'], axis=1)
Y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Augment the low class
os = SMOTE(random_state=42)
Aug_X, Aug_Y = os.fit_resample(X_train, y_train.ravel())

# train the model 
model = xgboost.XGBClassifier(random_state = 42)
model.fit(Aug_X, Aug_Y)

# train accuracy
pred_train = model.predict(X_train)
print("train accuracy: ", accuracy_score(pred_train, y_train))

# test accuracy
pred_test = model.predict(X_test)
print("test accuracy: ", accuracy_score(pred_test, y_test))

# confusion matrix 
df_test = X_test.copy()
df_test['prediction'] = pred_test
df_test['target'] = y_test
cm = df_test.groupby(['target', 'prediction'], as_index=False).size()
print(cm)

# save the model 
pickle.dump(model, open(model_filename, 'wb'))

