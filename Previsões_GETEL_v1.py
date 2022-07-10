#!/usr/bin/env python
# coding: utf-8

# In[174]:


# Parameters
telefonia = 'Fixa' # Específicar tipo da telefonia : 'Fixa' ou 'Móvel'
ramal = 'Ramais Digitais Básico' # Caso telefonia fixa, especificar o tipo de ramal
mes_inicio = 1 # Mês da primeira planilha 
ano_inicio = 2017 # Ano da primeira planilha
ano_fim = 2020 # Ano da última planilha
mes_fim = 12 # Mês da última planilha
num_prev = 24 # Quantidade, em meses, para a previsão


# In[175]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xlrd


# In[176]:


# Get the index of the type of ramal desired
def find_colum_index(ramal):
  ncol = inputWorksheet.ncols
  index = 0
  for i in range(ncol):
    if inputWorksheet.cell_value(1,i) == ramal:
      index = i
  if index == 0:
    print('Não encontrei esse tipo de ramal')
  else:
    return index


# In[177]:


# Get the sum of the ramal desired
def get_colum_sum(ramal):
  nrow = inputWorksheet.nrows
  total = 0  
  for i in range(nrow-2):
    # print(inputWorksheet.cell_value(i+2, find_colum_index(ramal)))
    if inputWorksheet.cell_value(i+2, find_colum_index(ramal)) != '':
        total += inputWorksheet.cell_value(i+2, find_colum_index(ramal))
  return total


# In[178]:


# Create time series
column_names = ['Mes', 'Ativos']
df_input = pd.DataFrame(columns= column_names)

for j in range(ano_inicio, ano_fim + 1):
    # print('============================')
    # print(j)
    # print('============================')
    for i in range(1,13):
        
       if telefonia == 'Fixa': 
           if (j == ano_inicio and i>= mes_inicio) or (j == ano_fim and i<= mes_fim) or (j!= ano_inicio and j!= ano_fim):
              string = str(i)
              if len(string) == 1:
                path = 'fixa/Total de PCMs com e sem PVF 0' + string + '_' + str(j) + '.xlsx'
              else:
                path = 'fixa/Total de PCMs com e sem PVF ' + string + '_' + str(j) + '.xlsx'
              inputWorkbook = xlrd.open_workbook(path)
              inputWorksheet = inputWorkbook.sheet_by_index(0)
              # print(path)
              df_input = df_input.append({'Mes': str(j) + '-' + str(i) + '-01','Ativos': int(get_colum_sum(ramal))}, ignore_index=True)
            
       elif telefonia == 'Móvel':
            
            if i == 1: 
              cur_month = 'JAN'
            elif i == 2: 
              cur_month = 'FEV'
            if i == 3: 
              cur_month = 'MAR'
            elif i == 4: 
              cur_month = 'ABR'
            if i == 5: 
              cur_month = 'MAI'
            elif i == 6: 
              cur_month = 'JUN'
            if i == 7: 
              cur_month = 'JUL'
            elif i == 8: 
              cur_month = 'AGO'
            if i == 9: 
              cur_month = 'SET'
            elif i == 10: 
              cur_month = 'OUT'
            if i == 11: 
              cur_month = 'NOV'
            elif i == 12: 
              cur_month = 'DEZ'
            
            if (j == ano_inicio and i>= mes_inicio) or (j == ano_fim and i<= mes_fim) or (j!= ano_inicio and j!= ano_fim):
              string = str(i)
              if len(string) == 1:
                path = 'Movel/0' + string + '_PLANTA_MOVEL_' + cur_month + str(j)[-2 :] + '.xls'
              else:
                path = 'Movel/' + string + '_PLANTA_MOVEL_' + cur_month + str(j)[-2 :] + '.xls'
              inputWorkbook = xlrd.open_workbook(path)
              inputWorksheet = inputWorkbook.sheet_by_name('PLANTA ' + cur_month + str(j)[-2 :])
              # print(path)
              df_input = df_input.append({'Mes': str(j) + '-' + str(i) + '-01','Ativos': int(inputWorksheet.nrows)}, ignore_index=True)
       else:
            print('Telefonia não encontrada')


# In[179]:


# Generate MLP input
column_names = ['Month1', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 'Y']
df = pd.DataFrame(columns= column_names)

for i in range(len(df_input)-6) :
  df = df.append({'Month1': df_input.iloc[i,1],'Month2': df_input.iloc[i+1,1], 'Month3': df_input.iloc[i+2,1], 'Month4': df_input.iloc[i+3,1], 'Month5': df_input.iloc[i+4,1], 'Month6': df_input.iloc[i+5,1], 'Y': df_input.iloc[i+6,1]}, ignore_index=True)

last_6_months = [df.iloc[len(df)-1, 1], df.iloc[len(df)-1, 2], df.iloc[len(df)-1, 3], df.iloc[len(df)-1, 4], df.iloc[len(df)-1, 5], df.iloc[len(df)-1, 6] ]
last_6_months


# In[180]:


# split the data
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=None)


# In[181]:


# trainning the neural network
regressor = MLPRegressor(hidden_layer_sizes=(200,200), activation='relu', max_iter = 5000, learning_rate_init=0.0001, early_stopping=False, n_iter_no_change = 1000, random_state=None)
regressor.fit(x_train, y_train)


# In[182]:


# score
y_pred = regressor.predict(x_test)
regressor.score(x_test, y_test)


# In[183]:


# standard deviation
sum_errs = np.sum((np.array(y_test)-np.array(y_pred))**2)
stdev = np.sqrt(1/(len(y_test)-2) * sum_errs)
stdev


# In[184]:


# Generate output excel file
column_names_df_output = ['Mes' , 'Ativos']

df_output = pd.DataFrame(columns= column_names_df_output)
df_output_low = pd.DataFrame(columns= column_names_df_output)
df_output_high = pd.DataFrame(columns= column_names_df_output)

if mes_fim < 12:
    month = mes_fim + 1
    year = ano_fim
elif mes_fim == 12:
    month = 1
    year = ano_fim + 1
    
dev = stdev

for i in range(1, num_prev):  
  new_instance = [last_6_months]
  predicted_value = regressor.predict(new_instance)   # predict next month value  
  
  y_max = predicted_value + 1.96*dev
  y_min = predicted_value - 1.96*dev  

  # print('Y max - Y min: ' + str(y_max - y_min))  

  df_output = df_output.append({'Mes': str(year) + '-' + str(month) + '-01','Ativos': int(predicted_value)}, ignore_index=True)
  df_output_low = df_output_low.append({'Mes': str(year) + '-' + str(month) + '-01','Ativos': int(y_min)}, ignore_index=True)
  df_output_high = df_output_high.append({'Mes': str(year) + '-' + str(month) + '-01','Ativos': int(y_max)}, ignore_index=True)
  last_6_months = [last_6_months[1],last_6_months[2], last_6_months[3], last_6_months[4], last_6_months[5], predicted_value[0]] # add new instance with the predicted value
  month +=1
      
  # dev = stdev*np.sqrt(1+(1/i))
  # dev = stdev*np.sqrt(1+(i/num_prev))
  dev = stdev*np.sqrt(i)
  # print(dev)  
    
  if month == 13:
        month =1
        year += 1
  #print('=========')
  #print(last_3_months)
  #dataset.loc[len(dataset)]=(dataset.iloc[len(dataset)-2, 0], dataset.iloc[len(dataset) - 2, 1],dataset.iloc[len(dataset)-2, 2] ,predicted_value)

# merge output dataframes
result = pd.concat([df_output_low, df_output, df_output_high], axis=1, join="inner")
# result = result.drop(index = 'Mes')
# del result[1]


result.columns = ['Mes', 'Ativos - Mínimo', 'Mes2', 'Ativos - Previsão', 'Mes3', 'Ativos - Máximo']
result = result.drop(result.columns[[2,4]], axis=1)

# export dataframe to excel
result.to_excel ('ativos.xlsx')
result


# In[185]:


# df_input = df_input.groupby('Mes')['Ativos'].sum().reset_index()
df_input['Mes'] = pd.to_datetime(df_input['Mes'])
df_input = df_input.set_index('Mes')

# df_output = df_output.groupby('Mes')['Ativos'].sum().reset_index()
df_output['Mes'] = pd.to_datetime(df_output['Mes'])
df_output = df_output.set_index('Mes')

# df_input = df_input.groupby('Mes')['Ativos'].sum().reset_index()
df_output_low['Mes'] = pd.to_datetime(df_output_low['Mes'])
df_output_low = df_output_low.set_index('Mes')

# df_input = df_input.groupby('Mes')['Ativos'].sum().reset_index()
df_output_high['Mes'] = pd.to_datetime(df_output_high['Mes'])
df_output_high = df_output_high.set_index('Mes')


# In[186]:


# df_input.plot(figsize=(15, 6))
# plt.show()
plt.figure(dpi =150)
plt.plot(df_input, 'b')
plt.plot(df_output, 'r')
plt.plot(df_output_high, '#808080')
plt.plot(df_output_low, '#808080')
plt.fill_between(df_output_low.index.to_numpy(), np.array(df_output_high['Ativos'].to_numpy(),dtype=float), np.array(df_output_low['Ativos'].to_numpy(),dtype=float), alpha = 0.5, color='#808080')
# df_output.plot(figsize=(15, 6),color = 'red')
plt.show()

#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

#ax2.fill_between(df_output_high, df_output_low, 1)
#ax2.set_ylabel('between y1 and 1')

print('A previsão para ' + str(num_prev) + " meses a frente é de: " + str(predicted_value))


# In[ ]:




