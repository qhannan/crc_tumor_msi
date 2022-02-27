import pandas as pd
from sklearn import preprocessing

msicsv = pd.read_csv('./msi_results.csv')
le = preprocessing.LabelEncoder()
le.fit(['MSI-H','MSI-L','MSS'])
msi_n = le.transform(msicsv['msi_results'])
msicsv['msi_results_n'] = msi_n
#msicsv.to_csv('./msiresults.csv')

allcase = msicsv['caseid'].unique()
