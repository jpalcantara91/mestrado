#!/usr/bin/env python
# coding: utf-8

# Inicialmente é necessário importar os dos dados coletados e dividi-lo em conjuntos de teste e validação 

# In[1]:


# !pip install python-mnist


# In[2]:


from mnist import MNIST
import random
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)


# In[3]:


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)


# In[4]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

print(x_train[0])


# In[5]:


num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)


# A partir de agora inicia a criação e treino da MLP

# In[6]:


# Parte 2 - Classificação dos sinais do Elevador
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

# define model
opt = Adam(learning_rate=0.001) # aqui é colocado a taxa de aprendizagem
                     # SGD indica que está usando a técnica de gradiente descendente estocástico para otimizar a MLP
                     # Essa técnica de otimização é a mais usada e é a que se aprende quando se lê sobre redes neurais 
model = Sequential()
model.add(Dense(784, input_dim=784, activation='sigmoid'))  # input_dim é o tamanho da entrada
#model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # o número '4' deve ser a quantidade de classes que a rede vai classificar
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=120)
# fit model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1, callbacks=[es], batch_size=10)
# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0) #usado para testar a acurácia do modelo
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
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


# In[7]:


predictions = model.predict(x_test)


# In[8]:


print(y_test[2])


# In[9]:


predictions[2]


# In[ ]:




