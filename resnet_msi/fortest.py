import pandas as pd
from sklearn import preprocessing

def removemsil(x):
    if x == 'MSI-H':
        y = 'MSI'
    else:
        y = 'MSS'
    return y

clean_df = pd.read_csv('source/test.csv')

le = preprocessing.LabelEncoder()
le.fit(['MSI','MSS'])
transloc = clean_df['msi_results'].apply(removemsil)
msi_n = le.transform(transloc)
clean_df['msi_label'] = msi_n
clean_df = clean_df.drop(columns=['msi_results_n'])
clean_df = clean_df.to_csv('target/test.csv',index=False)