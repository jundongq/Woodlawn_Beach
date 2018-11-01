import numpy as np 
import pandas as pd 
import xgboost as xgb

import sklearn
import h5py
import os
import getpass
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Import Data
DATA_DIR ='/Users/{}/Dropbox/VirtualBeach/Regression'.format(getpass.getuser())
train_eval_1 = pd.read_csv(os.path.join(DATA_DIR, 'Woodlawn_Jundong_2008-2016.csv'), header='infer', sep=',')
train_eval_2 = pd.read_csv(os.path.join(DATA_DIR, 'Woodlawn_Jundong_2017.csv'), header='infer', sep=',')
test = pd.read_csv(os.path.join(DATA_DIR, 'Woodlawn_Jundong_2018.csv'), header='infer', sep=',')

train_eval = pd.concat([train_eval_1, train_eval_2])

X_train = train_eval.iloc[:,3:46].copy()
y_train = train_eval.iloc[:,2].copy()
X_test = test.iloc[:,3:46].copy()
y_test = test.iloc[:,2].copy()
print type(X_train)
print type(X_test)
# Convert y_train and y_test to binary, 1:positive(unsafe), 0:negative(safe)
ecoli_threshold = 235.
y_train = np.asarray(map(lambda x: 1 if x>=np.log10(ecoli_threshold) else 0, y_train.tolist()))
y_test = np.asarray(map(lambda x: 1 if x>=np.log10(ecoli_threshold) else 0, y_test.tolist()))

#Randomize Input Data
random_seed = 6789
np.random.seed(random_seed)

perm = np.random.permutation(len(X_train))

# Randomly Shuffle the Dataset
X = X_train.iloc[perm]
y = y_train[perm]

val_size = 80

x_train = X.iloc[:np.shape(X)[0]-val_size, :]
x_val   = X.iloc[-val_size:, :]

y_train = y[:np.shape(X)[0]-val_size].tolist()
y_val   = y[-val_size:].tolist()
# print x_train.shape
# print x_val.shape

# Oversampling using SMOTE

sm = SMOTE(random_state=456)
X_res, y_res = sm.fit_sample(x_train, y_train) # return X_res, y_res as np.array

X_res_df = pd.DataFrame(data=X_res, columns = x_train.columns)

# x_train = X_res
# y_train = y_train
print x_train.shape
print x_val.shape
# Convert Input (np.array) into DMatrix

dtrain = xgb.DMatrix(X_res_df, label=y_res)
dval   = xgb.DMatrix(x_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)


param = {'max_depth':3, 'eta':0.002, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 1500
watchlist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)#, early_stopping_rounds=10)

# Make prediction

preds = bst.predict(dtest)
y_pred = map(lambda x: 1 if x>0.50 else 0, preds)
print preds
print accuracy_score(y_test, y_pred)
print classification_report(y_test, y_pred)

# Print confusion matrix
c_matrix = confusion_matrix(y_test, y_pred)
print c_matrix

# sns.heatmap(c_matrix, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=y_test, yticklabels=y_test)
# plt.xlabel('Predicted label')
# plt.ylabel('True label');
# plt.show()


"""
param = {'max_depth':4, 'eta':0.001, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 800

0.791666666667
             precision    recall  f1-score   support

          0       0.81      0.90      0.85        48
          1       0.74      0.58      0.65        24

avg / total       0.79      0.79      0.78        72


with SMOTE over_sampling
param = {'max_depth':3, 'eta':0.0025, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 1000
0.694444444444
             precision    recall  f1-score   support

          0       0.88      0.62      0.73        48
          1       0.53      0.83      0.65        24

avg / total       0.76      0.69      0.70        72


with SMOTE over_sampling
param = {'max_depth':3, 'eta':0.01, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 1500
0.805555555556
             precision    recall  f1-score   support

          0       0.89      0.81      0.85        48
          1       0.68      0.79      0.73        24

avg / total       0.82      0.81      0.81        72


"""
