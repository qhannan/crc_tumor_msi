import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read csvs
######################################################
#coad = pd.read_csv('source/msi_results_COAD.csv')
#read = pd.read_csv('source/msi_results_READ.csv')
#df = pd.concat([coad,read])
stad = pd.read_csv('source/msi_nanfang.csv')
df = stad
savepath = 'target/msi_crc_final.csv'
######################################################
# clean csvs
clean_df = pd.DataFrame(df[['bcr_patient_barcode','mononucleotide_and_dinucleotide_marker_panel_analysis_status']])
clean_df.rename(columns={'bcr_patient_barcode':'caseid','mononucleotide_and_dinucleotide_marker_panel_analysis_status':'msi_results'},inplace=True)
clean_df = clean_df[clean_df['msi_results']!='Indeterminate']


def removemsil(x):
    if x == 'MSI-H':
        y = 'MSI'
    else:
        y = 'MSS'
    return y
# label encoder
le = preprocessing.LabelEncoder()
le.fit(['MSI','MSS'])
transloc = clean_df['msi_results'].apply(removemsil)
print(transloc)
msi_n = le.transform(transloc)
clean_df['msi_label'] = msi_n
'''
allcases = clean_df['caseid'].unique()
temp_train, case_test = train_test_split(allcases, test_size=0.3, random_state = 11)
print(case_test)
case_train,case_val = train_test_split(temp_train, test_size=0.1, random_state = 11)

typelist = []
for i in clean_df['caseid']:
    if i in case_train:
        typelist.append('train')
    elif i in case_test:
        typelist.append('test')
    elif i in case_val:
        typelist.append('val')
print(len(typelist))
clean_df['type'] = typelist
'''
clean_df.to_csv(savepath,index=False)

