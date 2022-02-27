import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

test_results = pd.read_csv('target/crctestresults1.csv')
print(test_results.columns)

allcases = test_results['caseid'].unique()
print(allcases)

casename = []
caselabels = []
msirates = []
for case in allcases:
    tilesresults = test_results[test_results['caseid']==case]
    msinum = sum(tilesresults['predicts'])
    tilenum = len(tilesresults['predicts'])
    if tilenum > 0:
        msirate = msinum/tilenum
        caselabel = int(all(tilesresults['truelabels']))

        caselabels.append(caselabel)
        msirates.append(msirate)
        casename.append(case)
    else: 
        pass

y_pred = np.array(msirates)
y_true = np.array(caselabels)
aucc = roc_auc_score(y_true,y_pred)
print(aucc)
y_predn = (y_pred>0.5).astype('uint8')
auccc = accuracy_score(y_true,y_predn)
print(auccc)

aaa = confusion_matrix(y_true, y_predn)
print(aaa)

label = y_true
predict = y_pred

fpr,tpr,_ = roc_curve(label,predict)
roc_auc = auc(fpr,tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CRC-DX Patient-level ROC CURVE')
plt.legend(loc="lower right")
plt.show()
plt.savefig('./CRCROC.png')