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
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
import seaborn as sns

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

X = encode_categorical(X)
#censoring issues
n_censored = y.shape[0] - y["status"].sum()
print("%.1f%% of records are censored" % (n_censored / y.shape[0] * 100))

plt.figure(figsize=(9, 6))
val, bins, patches = plt.hist((y["time"][y["status"]],
                               y["time"][~y["status"]]),
                              bins=30, stacked=True)
plt.legend(patches, ["Time of Death", "Time of Censoring"])

for group in LUNG_CLM['ph.ecog'].unique():
    mask = X["ph.ecog"] == group
    time, surv_prob = kaplan_meier_estimator(
        y["status"][mask],
        y["time"][mask])

    plt.step(time, surv_prob, where="post",
             label="Treatment = {}".format(group))

estimator = FastSurvivalSVM(optimizer="rbtree", max_iter=1000, tol=1e-6, random_state=0)

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['status'], y['time'], prediction)
    return result[0]

param_grid = {'alpha': 2. ** np.arange(-12, 13, 2)}
cv = ShuffleSplit(n_splits=200, test_size=0.5, random_state=0)
gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                   n_jobs=4, iid=False, refit=False,
                   cv=cv)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
gcv = gcv.fit(X, y)
gcv.best_score_, gcv.best_params_

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















def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

scores = fit_and_score_features(X.values, y)
all_predictors=pd.Series(scores, index=X.columns).sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

pipe = Pipeline([('encode', OneHotEncoder()),
                 ('select', SelectKBest(fit_and_score_features, k=3)),
                 ('model', CoxPHSurvivalAnalysis())])

param_grid = {'select__k': np.arange(1, X.shape[1] + 1)}
gcv = GridSearchCV(pipe, param_grid, return_train_score=True)
gcv.fit(X, y)

pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
pipe.set_params(**gcv.best_params_)
pipe.fit(X, y)

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
best_predictors=pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])

pred_curves = estimator.predict_survival_function(X)
for curve in pred_curves:
    plt.step(curve.x, curve.y, where="post")

Xt = OneHotEncoder().fit_transform(X)

cv = KFold(n_splits=5, shuffle=True, random_state=312188)
coxnet = CoxnetSurvivalAnalysis(n_alphas=60,
                                l1_ratio=0.2, alpha_min_ratio=0.01,
                                verbose=True).fit(X_train, y_train)
gcv = GridSearchCV(coxnet,
                   {"alphas": [[v] for v in coxnet.alphas_]},
                   cv=cv).fit(X_train, y_train)

scores = gcv.cv_results_['mean_test_score']
scores_std = gcv.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
