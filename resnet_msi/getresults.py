import pandas as pd
import numpy as np
import os

# Something Input
###############################################################
resultsdir = '../detect_tumor/csvresutlss_nan0.5'
savepath = './target/crc_tumor_results.csv'
###############################################################

for i,csvfile in enumerate(os.listdir(resultsdir)):
    print(i)
    if i == 0:
        datacsv = pd.read_csv(os.path.join(resultsdir,csvfile))
        tumorcsv = datacsv[datacsv['label']==8]
    else:
        apcsv = pd.read_csv(os.path.join(resultsdir,csvfile))
        aptumorcsv = apcsv[apcsv['label']==8]
        tumorcsv = pd.concat([tumorcsv,aptumorcsv])

def get_case(x):
    return x[34:42]

filelist = tumorcsv['imgpath']
caselist = filelist.apply(get_case)
tumorcsv['caseid'] = caselist
print(caselist)
tumorcsv.to_csv(savepath,index = False)
