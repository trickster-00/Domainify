import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('RF_model.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

data = pd.read_csv('data/new_df.csv')

X = data.drop(["phishing"], axis=1)
y = data.phishing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 47)

model.fit(X_train,y_train)

pred = model.predict(X_test)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

logger.debug("Model results:")
logger.debug(f"Model score on traing dataset: {score_train}")
logger.debug(f"Model score on testing dataset: {score_test}")
logger.debug(f"Model's r2 score is: {metrics.r2_score(y_test,pred)}")

## Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

logger.debug(rf_random.best_params_)

prediction = rf_random.predict(X_test)

logger.debug("Model results after Hyperparameter Tuning:")
logger.debug(f"Model r2 score : {metrics.r2_score(y_test, prediction)}")
logger.debug(f"MAE score: {metrics.mean_absolute_error(y_test, prediction)}")
logger.debug(f"MSE score: {metrics.mean_squared_error(y_test, prediction)}")
