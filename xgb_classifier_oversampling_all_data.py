import numpy as np 
import pandas as pd 
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import sklearn
import h5py
import os
import getpass
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Import Data
DATA_DIR ='/Users/{}/Dropbox/VirtualBeach/Regression'.format(getpass.getuser())
train_eval_test = pd.read_csv(os.path.join(DATA_DIR, 'Woodlawn_2008_2018_16_features_target.csv'), header='infer', sep=',', index_col=0)
train_eval_test.reindex(range(len(train_eval_test)))

ecoli_threshold = 235.
X = train_eval_test.iloc[:,2:].copy()
y = train_eval_test.iloc[:,1].copy().apply(lambda x: 1 if x>np.log10(ecoli_threshold) else 0)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=456)
# X_train , X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.15, stratify=y_train_val, random_state=456)

# Oversampling using SMOTE
sm = SMOTE(random_state=456)
X_res, y_res = sm.fit_sample(X_train_val, y_train_val) # return X_res, y_res as np.array
X_res_df = pd.DataFrame(data=X_res, columns = X_train_val.columns)

print X_train_val.shape
print X_res.shape
# Convert Input (np.array) into DMatrix
# dtrain = xgb.DMatrix(X_res_df, label=y_res)
# dval   = xgb.DMatrix(X_val, label=y_val)
# dtest  = xgb.DMatrix(X_test, label=y_test)

# Using GridSearchCV to find best parameters
params = {
                'learning_rate': [0.001, 0.005, 0.01,  0.05, 0.1],
                'n_estimators': [1000, 1500, 2000],
                'max_depth': [3, 4, 5],
                'gamma': [i/10.0 for i in range(0,4)],
                'subsample': [i/10.0 for i in range(7,10)],
                'colsample_bytree': [i/10.0 for i in range(7,10)],
                'reg_alpha':[1e-5, 1e-2, 0.1, 1]        
}
gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5, \
                                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, \
                                        objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1, random_state=456), \
                                        param_grid = params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

gsearch.fit(X_res, y_res)

means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print "~~~~~~~~~~~~~~"
print gsearch.best_params_
print "~~~~~~~~~~~~~~"
print gsearch.best_score_

# Pick the best estimator through cross validation
bst = gsearch.best_estimator_

# Make prediction
y_pred = bst.predict(X_test.values)

print accuracy_score(y_test, y_pred)
print classification_report(y_test, y_pred)

# Print confusion matrix
c_matrix = confusion_matrix(y_test, y_pred)
print c_matrix

'''
# eval_set = [(X_train, y_train), (X_test, y_test)]
# eval_metric = ["auc","error"]
param = {'max_depth':4, 'eta':0.005, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', \
'subsample': '0.8', 'colsample_bytree': '0.5'}
num_round = 2000
watchlist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)#, early_stopping_rounds=10)

# Make prediction
preds = bst.predict(dtest)
y_pred = map(lambda x: 1 if x>0.50 else 0, preds)
# print preds
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
'''

