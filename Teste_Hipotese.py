#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.stats import ranksums
import pandas as pd
# excel 1
excel_1 = pd.read_excel('rwFSSnf_RC01.xlsx')
array_1 = []
array_2 = []
array_3 = [[0.0 for _ in range(25)] for i in range(10)]
# excel 2
excel_2 = pd.read_excel('rwFSSnfStpInd_RC01.xlsx')
array_4 = []
array_5 = []
array_6 = [[0.0 for _ in range(25)] for i in range(10)]
for i in range(10):
    # excel 1
    array_1.append(excel_1.iloc[i,20])
    array_2.append(array_1[i].split(','))
    array_2[i][0] = array_2[i][0].replace('[','')
    array_2[i][24] = array_2[i][24].replace(']','')
    # excel 2
    array_4.append(excel_2.iloc[i,20])
    array_5.append(array_4[i].split(','))
    array_5[i][0] = array_5[i][0].replace('[','')
    array_5[i][24] = array_5[i][24].replace(']','')
    for j in range(25):
        # excel 1
        array_3[i][j] = float(array_2[i][j])
        array_6[i][j] = float(array_5[i][j])
        # excel 2
for i in range(10): 
    b = ranksums(array_3[i],array_6[i], alternative = 'less')
    print('para o experimento ' + str(i) + ' o valor do pValue foi de ' + str(b[1]))


# In[ ]:




