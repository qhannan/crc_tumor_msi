import pandas as pd
import numpy as np
tumorpath = 'target/crc_tumor_results.csv'
msipath = 'target/msi_crc_final.csv'
savepath = 'target/crc_final.csv'

tumor = pd.read_csv(tumorpath)
print(len(tumor))
msi = pd.read_csv(msipath)
print(len(msi))

final = pd.merge(tumor,msi,how='left')
final = final[np.isnan(final['msi_label'])==False]
final.to_csv(savepath,index=False)
print(len(final))
'''
train = final[final['type']=='train']
train.to_csv('{}_train.csv'.format(savepath[:-4]))
val = final[final['type']=='val']
val.to_csv('{}_val.csv'.format(savepath[:-4]))
test = final[final['type']=='test']
test.to_csv('{}_test.csv'.format(savepath[:-4]))
'''
