"""
Created on Mon Mar  4 12:52:25 2019

@author: User
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit
from sksurv.column import encode_categorical
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
import seaborn as sns
import warnings

LUNG_LOC = r"/home/ed/Desktop/lung_cancer/lung.csv"
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

X = encode_categorical(X)
#censoring issues
n_censored = y.shape[0] - y["status"].sum()
print("%.1f%% of records are censored" % (n_censored / y.shape[0] * 100))

for sex in LUNG_CLM['sex'].unique():
    msk = LUNG_CLM['sex'] ==sex    
    plt.figure(figsize=(9, 6))
    val, bins, patches = plt.hist((y["time"][msk][y["status"][msk]],
                                   y["time"][msk][~y["status"][msk]]),
                                  bins=30, stacked=True)
    plt.legend(patches, ["Time of Death", "Time of Censoring"])
#%%
for group in LUNG_CLM['ph.ecog'].unique():
    mask = X["ph.ecog"] == group
    time, surv_prob = kaplan_meier_estimator(
        y["status"][mask],
        y["time"][mask])

    plt.step(time, surv_prob, where="post",
             label="Treatment = {}".format(group))
#%%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,test_size=0.2,
                                                   random_state=4)

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['status'], y['time'], prediction)
    return result[0]
warnings.filterwarnings("once")
#optimizers=("rbtree""avltree","direct-count","PRSVM","rbtree","simple"

estimator = FastSurvivalSVM(optimizer="rbtree",random_state=1234)
param_grid = {'alpha': 2. ** np.arange(-13, 10, 2), 'tol': (1e+2, 1e-5), 'max_iter': (90, 100)}
cv = ShuffleSplit(n_splits=200, test_size=0.2, random_state=0)
gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                   n_jobs=4, iid=False, refit=False,
                   cv=cv)

gcv = gcv.fit(X, y)
print(gcv.best_score_, gcv.best_params_)


def plot_performance(gcv):
    n_splits = gcv.cv.n_splits
    cv_scores = {"alpha": [], "test_score": [], "split": []}
    order = []
    for i, params in enumerate(gcv.cv_results_["params"]):
        name = "%.5f" % params["alpha"]
        order.append(name)
        for j in range(n_splits):
            vs = gcv.cv_results_["split%d_test_score" % j][i]
            cv_scores["alpha"].append(name)
            cv_scores["test_score"].append(vs)
            cv_scores["split"].append(j)
    df = pd.DataFrame.from_dict(cv_scores)
    _, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x="alpha", y="test_score", data=df, order=order, ax=ax)
    _, xtext = plt.xticks()
    for t in xtext:
        t.set_rotation("vertical")

plot_performance(gcv)
refit_gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                         n_jobs=4, iid=False, refit=True,
                         cv=cv)

refit_gcv = refit_gcv = refit_gcv.fit(X, y)
gridsearch_rank = refit_gcv.predict(X)
#%%
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.kernels import clinical_kernel

kernel_matrix = clinical_kernel(X)
kssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=1234)

kgcv = GridSearchCV(kssvm, param_grid, score_survival_model,
                    n_jobs=4, iid=False, refit=False,
                    cv=cv)


warnings.filterwarnings("ignore", category=UserWarning)
kgcv = kgcv.fit(kernel_matrix, y)

print(kgcv.best_score_, kgcv.best_params_)

plot_performance(kgcv)

refit_kgcv = GridSearchCV(kssvm, param_grid, score_survival_model,
                          n_jobs=6, iid=False, refit=True,
                          cv=cv)
refit_kgcv = refit_kgcv.fit(kernel_matrix, y)
fast_kernel_SSVM_gcvrank = refit_kgcv.predict(kernel_matrix)



X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,test_size=0.2,
                                                   random_state=4)
best_param=refit_kgcv.best_params_

estimator = CoxPHSurvivalAnalysis(**refit_kgcv.best_params_)
estimator.fit(X_train, y_train)
estimator.score(X_test, y_test)


pred_curves = estimator.predict_survival_function(X_test)
for curve in pred_curves:
    plt.step(curve.x, curve.y, where="post")
    
plt.clf()

cum_curv = estimator.predict_cumulative_hazard_function(X_test)
for item in cum_curv:
    plt.step(item.x, item.y, where="post")

#%%
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['status'], y['time'], prediction)
    return result[0]
#%%
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
model3=ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', 
                                              learning_rate=0.2, 
                                              n_estimators=1000, 
                                              subsample=1.0, 
                                              dropout_rate=0, 
                                              random_state=123, 
                                              verbose=1)

model3.fit(X_train, y_train)
model3.predict(X_test)
model3.score(X_train, y_train)
#%%
LUNG_LOC = r"/home/ed/Desktop/lung_cancer/lung.csv"
LUNG_CLM = pd.read_csv(LUNG_LOC)
LUNG_CLM.fillna(0, inplace=True)
LUNG_CLM['status'] = LUNG_CLM[['status']] == 2

X = LUNG_CLM[['inst',
              'age',
              'sex',
              'ph.ecog',
              'ph.karno',
              'pat.karno',
              'meal.cal',
              'wt.loss',
              'status',
              'time']]
X['xg_time']=X.time
msk = X['status'] == 0
X.loc[msk,'xg_time']=X.time * -1

y=X.xg_time
X = LUNG_CLM[['inst',
              'age',
              'sex',
              'ph.ecog',
              'ph.karno',
              'pat.karno',
              'meal.cal',
              'wt.loss']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,test_size=0.2,
                                                   random_state=1234)


import xgboost as xgb
from xgboost import XGBClassifier
model4=XGBClassifier(max_depth=30, 
              learning_rate=0.1, 
              n_estimators=1000, 
              verbosity=1, 
              silent=None, 
              objective='survival:cox', 
              booster='gblinear', 
              n_jobs=10, 
              nthread=-1, 
              gamma=0, 
              min_child_weight=1,
              max_delta_step=0, 
              subsample=1, 
              colsample_bytree=1, 
              colsample_bylevel=1, 
              colsample_bynode=1, 
              reg_alpha=0,
              reg_lambda=0, 
              scale_pos_weight=1, 
              base_score=100, 
              random_state=100, 
              seed=1234, 
              missing=True)

model4.fit(X_test, y_test, 
           sample_weight=None,
           eval_set=None,
           eval_metric='rmse',
           early_stopping_rounds=None,
           verbose=True,
#          xgb_model=None,
           sample_weight_eval_set=None)

t=model4.predict(X_train)
model4.score(X_train, y_train)
#xgb.plot_importance(model4, importance_type='weight',max_num_features=10)
#coeff=model4.coef_
