"""
Created on Mon Mar  4 12:52:25 2019

@author: User
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from sksurv.linear_model import CoxPHSurvivalAnalysis

LUNG_LOC = r"/home/ed/Desktop/lung.csv"
LUNG_CLM = pd.read_csv(LUNG_LOC)
LUNG_CLM.fillna(0, inplace=True)
LUNG_CLM['status'] = LUNG_CLM[['status']] == 2

STGE = LUNG_CLM[['status', 'time']]

y = STGE.to_records(index=False)

X = LUNG_CLM[['inst',
              'age',
              'sex',
              'ph.ecog',
              'ph.karno',
              'pat.karno',
              'meal.cal',
              'wt.loss']]

for group in LUNG_CLM.sex.unique():
    mask = X["sex"] == group
    time, surv_prob = kaplan_meier_estimator(
        y["status"][mask],
        y["time"][mask])

    plt.step(time, surv_prob, where="post",
             label="Treatment = {}".format(group))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ENCODER = OneHotEncoder()
ESTIMATOR = CoxPHSurvivalAnalysis()
ESTIMATOR.fit(ENCODER.fit_transform(X_train), y_train)

data_new_raw = pd.DataFrame(X_test)
data_new = ENCODER.transform(data_new_raw)

pred_curves = ESTIMATOR.predict_survival_function(data_new)
for curve in pred_curves:
    plt.step(curve.x, curve.y, where="post")

Xt = OneHotEncoder().fit_transform(X_train)

cv = KFold(n_splits=5, shuffle=True, random_state=328)
coxnet = CoxnetSurvivalAnalysis(n_alphas=1000,
                                l1_ratio=0.05, alpha_min_ratio=0.01,
                                verbose=True).fit(X_train, y_train)

gcv = GridSearchCV(coxnet,
                   {"alphas": [[v] for v in coxnet.alphas_]},
                   cv=cv).fit(X_train, y_train)

scores = gcv.cv_results_['mean_test_score']
scores_std = gcv.cv_results_['std_test_score']
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alphas, scores)
