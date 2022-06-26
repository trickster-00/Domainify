import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('model.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)

import pandas as pd
import numpy as np

data = pd.read_csv('domain')
data.drop('Unnamed: 0', axis=1, inplace=True)

x = data.iloc[:,:5]
y = data.drop(x, axis=1)

from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=35)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Support vector machines
from sklearn.svm import SVC
svmodel = SVC(kernel='linear', probability=True)
svmodel.fit(X_train,y_train)
ytrain_pred = svmodel.predict_proba(X_train)
logger.debug("SVM Classifier")
logger.debug('RF train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = svmodel.predict_proba(X_test)
logger.debug('RF test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#Random forests
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
ytrain_pred = rf_model.predict_proba(X_train)
f=("Random forests Classifier")
d=('RF train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = rf_model.predict_proba(X_test)
u=('RF test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_classifier=LogisticRegression()
log_classifier.fit(X_train, y_train)
ytrain_pred = log_classifier.predict_proba(X_train)
print('Logistic train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = log_classifier.predict_proba(X_test)
print('Logistic test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
ada_classifier=AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
ytrain_pred = ada_classifier.predict_proba(X_train)
logger.debug("Adaboost Classifier")
logger.debug('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = ada_classifier.predict_proba(X_test)
logger.debug('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#KNNClassifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
ytrain_pred = knn_classifier.predict_proba(X_train)
logger.debug("KNNClassifier")
logger.debug('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = knn_classifier.predict_proba(X_test)
logger.debug('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#max accuracy
pred=[]
for model in [svmodel,rf_model,log_classifier,ada_classifier,knn_classifier]:
    pred.append(pd.Series(model.predict_proba(X_test)[:,1]))
final_prediction=pd.concat(pred,axis=1).mean(axis=1)
logger.debug("Emsemble learing results")
logger.debug('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_prediction)))

# Calculate the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, final_prediction)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_prediction > thres, 1, 0)
    accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))

accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)], axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
logger.debug("Highest Threshold & Accuracy score")
logger.debug(accuracy_ls.head(3))

import pickle
file = open("rf_pred", 'wb')
pickle.dump(rf_model, file)
