import pandas as pd

train = pd.read_csv('target/crc_final_train.csv')
val = pd.read_csv('target/crc_final_val.csv')
print(len(train))
print(len(val))
#savetrainpath = 'target/crc_finaltrain_2.csv'
#savevalpath = 'target/crc_finalval_2.csv'

trainmss = train[train['msi_label']==1]
trainmsi = train[train['msi_label']==0]

valmss = val[val['msi_label']==1]
valmsi = val[val['msi_label']==0]

def balance(msi,mss,name,ratio):
    msi_num = msi.shape[0]
    mss_num = mss.shape[0]
    print(msi_num,mss_num)
    min_num = min(msi_num,mss_num)
    mss_train = mss.sample(n=min_num*ratio)#,random_state=11)
    msi_train = msi.sample(n=min_num)#,random_state=11)

    final = pd.concat([mss_train,msi_train])
    final.to_csv('target/crc_final{}{}.csv'.format(name,ratio),index=False)

balance(trainmsi,trainmss,'train',2)
balance(valmsi,valmss,'val',2)

'''
dic = {'a':[1, 2, 3, 4], 'b':[5, 6, 7, 8],
'c':[9, 10, 11, 12], 'd':[13, 14, 15, 16]}
df1=pd.DataFrame(dic)
print(df1)
df2=df1.sample(frac=0.5)
print(df2)
rowlist=[]
for indexs in df2.index:
    rowlist.append(indexs)
df3=df1.drop(rowlist,axis=0)
print(df3)
'''

