"""
Created on Mon Mar  4 12:52:25 2019

@author: User
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc

lung_loc=r"C:\Users\User\Desktop\lung.csv"

lung_clm=pd.read_csv(lung_loc)
lung_clm.fillna(0,inplace=True)
lung_clm['status']=lung_clm[['status']]==2
msk=lung_clm.time<=300
lung_clm['RSK1']=0
lung_clm.loc[msk,'RSK1']=1

stge=lung_clm[['status','time']]

y = stge.to_records(index=False)
                     
X=lung_clm[['inst',
 'age',
 'sex',
 'ph.ecog',
 'ph.karno',
 'pat.karno',
 'meal.cal',
 'wt.loss']]
          
from sksurv.nonparametric import kaplan_meier_estimator

for group in (lung_clm.sex.unique()):
    mask = X["sex"] == group
    time, surv_prob = kaplan_meier_estimator(
        y["status"][mask],
        y["time"][mask])

    plt.step(time, surv_prob, where="post",
             label="Treatment = {}".format(group))

from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder()
estimator = CoxPHSurvivalAnalysis()
estimator.fit(encoder.fit_transform(X_train), y_train)

data_new_raw = pd.DataFrame(X_test)
data_new = encoder.transform(data_new_raw)

pred_curves = estimator.predict_survival_function(data_new)
for curve in pred_curves:
    plt.step(curve.x, curve.y, where="post")
    
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold

Xt = OneHotEncoder().fit_transform(X_train)

cv = KFold(n_splits=5, shuffle=True, random_state=328)
coxnet = CoxnetSurvivalAnalysis(n_alphas=1000,
    l1_ratio=1.0, alpha_min_ratio=0.01).fit(X_train, y_train)

gcv = GridSearchCV(coxnet,
    {"alphas": [[v] for v in coxnet.alphas_]},
    cv=cv).fit(X_train, y_train)

scores = gcv.cv_results_['mean_test_score']
scores_std = gcv.cv_results_['std_test_score']
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alphas, scores)

