import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('logistic.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

data = pd.read_csv('data/new_df.csv')

X = data.drop('phishing', axis=1)
y= data.phishing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=44)

model.fit(X_train,y_train)

pred = model.predict(X_test)

logger.debug('Model predictions')
logger.debug(f"r2 score: {metrics.r2_score(y_test,pred)}")
logger.debug(f"MAE score: {metrics.mean_absolute_error(y_test,pred)}")

# Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

param_grid = [
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [1000]
    }
]

cv = RandomizedSearchCV(estimator=model, param_distributions= param_grid, scoring='neg_mean_squared_error', n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = 1)

best_cv = cv.fit(X_train, y_train)

prediction = best_cv.predict(X_test)

best_cv.best_estimator_

logger.debug('Results after Hyperparameter tuning')
logger.debug(f'Accuracy: {best_cv.score(X_test,y_test)}')
logger.debug(f'r2 score: {best_cv.r2_score(X_test,y_test)}')