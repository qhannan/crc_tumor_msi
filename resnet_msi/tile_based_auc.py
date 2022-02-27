import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt



df = pd.read_csv('source/staddxtestresults2.csv')
output0 = df['output0']
output1 = df['output1']
outputs = output1-output0

accpredicts = df['predicts']
acclabels = df['truelabels']
acc = accuracy_score(acclabels,accpredicts)
print(acc)


predict = outputs
label = df['msi_label']

fpr,tpr,_ = roc_curve(label,predict)
roc_auc = auc(fpr,tpr)


fpr_micro, tpr_micro, _ = roc_curve(label.ravel(), predict.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig('./stad_tiles-based-auc.png')