import fileio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import numpy as np
from matplotlib import pyplot as plt

trainX = fileio.pklOpener('trainSetDF.pkl')
testX = fileio.pklOpener('testSetDF.pkl')
trainY = trainX['SepsisLabel']
testY = testX['SepsisLabel']
trainX.drop(['SepsisLabel', 'Filename'], axis=1, inplace=True)
testX.drop(['SepsisLabel', 'Filename'], axis=1, inplace=True)

dictionaryroc = {}
dictionarypr = {}

cols = list(trainX.columns)
for column in cols:
    clf = LogisticRegression(max_iter=250, class_weight='balanced').fit(np.array(trainX[column]).reshape(-1, 1), trainY)
    out_prob = clf.predict_proba(np.array(testX[column]).reshape(-1, 1))
    auc = roc_auc_score(np.array(testY).reshape(-1, 1), out_prob[:, 1])
    ap = average_precision_score(np.array(testY).reshape(-1, 1), out_prob[:, 1])
    dictionaryroc[column] = auc
    dictionarypr[column] = ap

figroc = plt.figure(1, figsize=(10, 5))
figpr = plt.figure(2, figsize=(10, 5))

plt.figure(1)
plt.bar(list(dictionaryroc.keys()), list(dictionaryroc.values()))
plt.xticks(rotation=90, fontsize=6)
plt.xlabel("Feature")
plt.ylabel("AUROC w/ Logistic Regression")
figroc.savefig('uniAUROC')

plt.figure(2)
plt.bar(list(dictionarypr.keys()), list(dictionarypr.values()), color='red')
plt.xticks(rotation=90, fontsize=6)
plt.xlabel("Feature")
plt.ylabel("AUPRC w/ Logistic Regression")
figpr.savefig('uniAUPRC')
