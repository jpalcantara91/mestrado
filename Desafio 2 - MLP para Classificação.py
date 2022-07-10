#!/usr/bin/env python
# coding: utf-8

# Inicialmente é necessário importar os dos dados coletados e dividi-lo em conjuntos de teste e validação 

# In[1]:


# Recebe os CSVs dos novos sinais gerados, divide em banco de dados de treino e teste, cria os bancos de dados
# de entradas e labels integrados e com todos os valores gerados para treinar a MLP  
import random
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lenght_signal = 850 # Precisa ser o mesmo número usado no programa "Desafio 2 - Gerador de Sinais"
tamanho_test = 0.2 # tamanho do conjunto de teste em porcentagem

#Parte 1 - Recepção dos dados e separação dos conjuntos de teste e validação de cada classe
# ENTRADA da Classe 0
dataset = pd.read_csv("Sinal do elevador correto.csv") 
dataset = dataset.drop(columns =['Unnamed: 0'])
train = dataset.drop(columns =[str(lenght_signal)]) 
label = dataset[[str(lenght_signal)]]
X_train0, X_test0, y_train0, y_test0 = train_test_split(train, label, test_size=tamanho_test) # separa 20% dos valores para validação.  A vantagem de separar os dados
                                                                                    # de validação por classe garante uma distribuição homogênea para treinamento
                                                                                    
# ENTRADA da Classe 1
dataset = pd.read_csv("Sinal do elevador falha_acel.csv")
dataset = dataset.drop(columns =['Unnamed: 0'])
train = dataset.drop(columns =[str(lenght_signal)]) 
label = dataset[[str(lenght_signal)]]
X_train1, X_test1, y_train1, y_test1 = train_test_split(train, label, test_size=tamanho_test) # separa 20% dos valores para validação 

# ENTRADA da Classe 2
dataset = pd.read_csv("Sinal do elevador falha_vel.csv") 
dataset = dataset.drop(columns =['Unnamed: 0'])
train = dataset.drop(columns =[str(lenght_signal)]) 
label = dataset[[str(lenght_signal)]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(train, label, test_size=tamanho_test) # separa 20% dos valores para validação 

# ENTRADA da Classe 3
dataset = pd.read_csv("Sinal do elevador ruido_desl.csv")
dataset = dataset.drop(columns =['Unnamed: 0'])
train = dataset.drop(columns =[str(lenght_signal)]) 
label = dataset[[str(lenght_signal)]]
X_train3, X_test3, y_train3, y_test3 = train_test_split(train, label, test_size=tamanho_test) # separa 20% dos valores para validação 

# Junta todas as classes de entradas em um só banco de dados de treino
new_sheet = pd.concat([X_train0, X_train1, X_train2, X_train3], axis=0,sort=False) 
new_sheet_list = pd.concat([y_train0, y_train1, y_train2, y_train3], axis=0,sort=False)
new_sheet[lenght_signal] = new_sheet_list # coloca as labels na última coluna para embralhar no comando seguinte
new_sheet_shuffled = shuffle(new_sheet) # embaralha as entradas para deixar o treinamento da MLP otimizado com menos chance de 
                                # de se prender a mínimos locais
trainX = new_sheet_shuffled.drop(columns =[lenght_signal]) # conjunto de treino
trainy = new_sheet_shuffled[[lenght_signal]]    # labels do conjunto de treino


# Junta todas as classes de entradas em um só banco de dados de validação
testX = pd.concat([X_test0, X_test1, X_test2, X_test3], axis=0,sort=False) # conjunto de validação
testy = pd.concat([y_test0, y_test1, y_test2, y_test3], axis=0,sort=False) # labels do conjunto de validação


#Plot a row of X_train
#row = X_train.iloc[1]
#plt.plot(row)

# Salva os dados gerados em CSVs
trainX.to_csv('input_train.csv',  index=False)
trainy.to_csv('label_input_train.csv',  index=False)
testX.to_csv('input_validation.csv',  index=False)
testy.to_csv('label_input_validation.csv',  index=False)


# A partir de agora inicia a criação e treino da MLP

# In[2]:


# Parte 2 - Classificação dos sinais do Elevador
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# define model
opt = SGD(learning_rate=0.001) # aqui é colocado a taxa de aprendizagem
                     # SGD indica que está usando a técnica de gradiente descendente estocástico para otimizar a MLP
                     # Essa técnica de otimização é a mais usada e é a que se aprende quando se lê sobre redes neurais 
model = Sequential()
model.add(Dense(850, input_dim=850, activation='sigmoid'))  # input_dim é o tamanho da entrada
#model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax')) # o número '4' deve ser a quantidade de classes que a rede vai classificar
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1, callbacks=[es], batch_size=10)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0) #usado para testar a acurácia do modelo
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()


# In[3]:


predictions = model.predict(testX)


# In[11]:


predictions[100]


# In[ ]:




