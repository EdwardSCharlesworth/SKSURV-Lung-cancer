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
from sklearn.metrics import average_precision_score, make_scorer, confusion_matrix
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
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

def score_survival_model(model, x, y):
    prediction = model.predict(x)
    result = concordance_index_censored(y['status'], y['time'], prediction)
    return result[0]
warnings.filterwarnings("ignore")
#optimizers=("rbtree")#"avltree","direct-count","prsvm","rbtree","simple"

estimator = FastSurvivalSVM(optimizer="rbtree",random_state=1234)
param_grid = {'alpha': 2. ** np.arange(-6, 6, 2), 'tol': (.1, 1e-10), 'max_iter': (2, 400)}
cv = ShuffleSplit(n_splits=200, test_size=0.25, random_state=0)
gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                   n_jobs=4, iid=False, refit=False,
                   cv=cv)

gcv = gcv.fit(X, y)
#print(m)
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
estimator = GradientBoostingSurvivalAnalysis(loss='coxph',criterion='friedman_mse',
                                             min_impurity_split=None, 
                                             min_impurity_decrease=0.0, 
                                             random_state=None, max_features=None, 
                                             max_leaf_nodes=None, presort='auto', 
                                             subsample=1.0, dropout_rate=0.0, verbose=1 )
param_grid= {'learning_rate':(.1,0.2,0.3,0.4),
             'n_estimators':(100,200,300,400),
             'min_samples_split':(2,3,4,5,6), 
             'min_samples_leaf':(1,2,3,4,5),
             'min_weight_fraction_leaf':(0.0,0.1,0.2,0.3,0.4),
             'max_depth':(5,10,20,30,40)} 
                               
gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                   n_jobs=4, iid=False, refit=True,
                   cv=cv)

gcv = gcv.fit(X, y)
#print(m)
print(gcv.best_score_, gcv.best_params_)

estimator.fit(X_train, y_train)
estimator.score(X_test, y_test)


pred_curves = estimator.predict_survival_function(X_test)
for curve in pred_curves:
    plt.step(curve.x, curve.y, where="post")
    
plt.clf()

cum_curv = estimator.predict_cumulative_hazard_function(X_test)
for item in cum_curv:
    plt.step(item.x, item.y, where="post")

