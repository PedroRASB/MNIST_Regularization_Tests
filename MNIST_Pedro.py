
# coding: utf-8

# # Exemplo de rede neural MLP e Convolucional com Keras

# ## Importando as bibliotecas

# In[1]:


import keras


# In[2]:


from keras.datasets import mnist


# ## Fazendo o download dos dados

# In[3]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# ## Analisando os dados

# In[4]:


import matplotlib.pyplot as plt


# In[5]:


train_images.shape


# In[6]:


len(train_labels)


# In[7]:


train_images[0]


# In[8]:


plt.imshow(train_images[0], cmap='gray')


# In[9]:


train_labels


# In[10]:


test_images.shape


# In[11]:


len(test_labels)


# In[12]:


plt.imshow(test_images[0], cmap='gray')


# In[13]:


test_labels


# ## Normalizando os dados

# In[14]:


train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


# In[15]:


train_images[0]


# ### Entrada da MLP

# In[16]:


train_images_mlp = train_images.reshape((60000, 28 * 28))
test_images_mlp = test_images.reshape((10000, 28 * 28))


# In[17]:


train_images_mlp.shape


# ### Entrada da rede convolucional

# In[18]:


from keras import backend as K


# In[19]:


img_rows = 28
img_cols = 28

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[20]:


print('input shape:', input_shape)


# ### Transformando rótulos em dados categóricos

# In[21]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[22]:


train_labels[0]


# model: https://keras.io/models/model/
# 
# layers: https://keras.io/layers/about-keras-layers/

# In[23]:


from keras import models
from keras import layers


# # Rede 1: MLP 2 camadas

# In[24]:


#definindo a rede 
#model: https://keras.io/models/model/
#layers: https://keras.io/layers/about-keras-layers/

network1 = models.Sequential()
network1.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network1.add(layers.Dense(10, activation='softmax'))


# In[25]:


network1.summary()


# In[26]:


#paraleliza o modelo para treinamento em 2 GPUs
network1 = keras.utils.multi_gpu_model(network1,gpus=2)


# In[27]:


#compilando e treinando rede
#optimizers: https://keras.io/optimizers/
#loss funcitons: https://keras.io/losses/
#metrics: https://keras.io/metrics/

network1.compile(optimizer='sgd',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history_mlp = network1.fit(train_images_mlp, train_labels, epochs=140, batch_size=128, validation_split=0.2)


# In[28]:


#funcao que avalia a rede e retorna seus erros
def Avalia(hist,net,test_img,test_lab, i):#i:usado com early stopping, para selecionar a rede certa, 1 se nao for usado early stopping
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'b--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'b--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    test_loss, test_acc = net.evaluate(test_img, test_lab)
    print('test acc=', test_acc)
    print('training accuracy=',hist.history['acc'][-i])
    print('validation accuracy=',hist.history['val_acc'][-i])
    print('test err=', test_loss)
    print('training err=',hist.history['loss'][-i])
    print('validation err=',hist.history['val_loss'][-i])


# In[29]:


Avalia(history_mlp,network1,test_images_mlp,test_labels,1)


# # Rede 2: Convolucional 7 camadas

# In[31]:


#definindo rede
network2 = models.Sequential()
network2.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
network2.add(layers.Conv2D(64, (3, 3), activation='relu'))
network2.add(layers.Conv2D(64, (3, 3), activation='relu'))
network2.add(layers.Conv2D(64, (3, 3), activation='relu'))
network2.add(layers.Flatten())
network2.add(layers.Dense(128, activation='relu'))
network2.add(layers.Dense(64, activation='relu'))
network2.add(layers.Dense(10, activation='softmax'))


# In[32]:


network2.summary()


# In[33]:


network2 = keras.utils.multi_gpu_model(network2,gpus=2) #paralelizando a rede para treinamento em 2 GPUs


# In[34]:


#Compilando rede
network2.compile(optimizer='sgd',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])


# In[35]:


#treinando rede
history = network2.fit(train_images, train_labels,
                      batch_size=128,
                      epochs=140,
                      validation_split=0.2)


# In[36]:


#avaliando rede
Avalia(history,network2,test_images,test_labels,1)


# # Rede 3: Convolucional com MaxPooling

# In[38]:


#Definindo rede
network3 = models.Sequential()
network3.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
network3.add(layers.Conv2D(64, (3, 3), activation='relu'))
network3.add(layers.Conv2D(64, (3, 3), activation='relu'))
network3.add(layers.MaxPooling2D(pool_size=(2, 2)))
network3.add(layers.Conv2D(64, (3, 3), activation='relu'))
network3.add(layers.MaxPooling2D(pool_size=(2, 2)))
network3.add(layers.Flatten())
network3.add(layers.Dense(128, activation='relu'))
network3.add(layers.Dense(64, activation='relu'))
network3.add(layers.Dense(10, activation='softmax'))

network3.summary()


# In[39]:


#compilando e treinando
network3 = keras.utils.multi_gpu_model(network3,gpus=2)

network3.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])

history3=network3.fit(train_images, train_labels,
                      batch_size=128,
                      epochs=140,
                      validation_split=0.2)


# In[40]:


#avaliando rede
Avalia(history3,network3,test_images,test_labels,1)


# # Rede 4: Convolucional com MaxPooling e Dropout

# In[42]:


#Definindo rede
network4 = models.Sequential()
network4.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',input_shape=input_shape))
network4.add(layers.Conv2D(64, (3, 3), activation='relu'))
network4.add(layers.Conv2D(64, (3, 3), activation='relu'))
network4.add(layers.MaxPooling2D(pool_size=(2, 2)))
network4.add(layers.Conv2D(64, (3, 3), activation='relu'))
network4.add(layers.MaxPooling2D(pool_size=(2, 2)))
network4.add(layers.Dropout(0.25))
network4.add(layers.Flatten())
network4.add(layers.Dense(128, activation='relu'))
network4.add(layers.Dropout(0.5))
network4.add(layers.Dense(64, activation='relu'))
network4.add(layers.Dropout(0.5))
network4.add(layers.Dense(10, activation='softmax'))

network4.summary()


# In[43]:


#compilando e treinando
network4 = keras.utils.multi_gpu_model(network4,gpus=2)

network4.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])

history4=network4.fit(train_images, train_labels,
                      batch_size=128,
                      epochs=140,
                      validation_split=0.2)


# In[44]:


#avaliando rede
Avalia(history4,network4,test_images,test_labels,1)


# # Rede 5: Convolucional com MaxPooling, Dropout e Weight Decay (norma L2)

# In[46]:


#Definindo rede
network5 = models.Sequential()
network5.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',input_shape=input_shape))
network5.add(layers.Conv2D(64, (3, 3), activation='relu'))
network5.add(layers.Conv2D(64, (3, 3), activation='relu'))
network5.add(layers.MaxPooling2D(pool_size=(2, 2)))
network5.add(layers.Conv2D(64, (3, 3), activation='relu'))
network5.add(layers.MaxPooling2D(pool_size=(2, 2)))
network5.add(layers.Dropout(0.25))
network5.add(layers.Flatten())
network5.add(layers.Dense(128, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network5.add(layers.Dropout(0.5))
network5.add(layers.Dense(64, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network5.add(layers.Dropout(0.5))
network5.add(layers.Dense(10, activation='softmax'))

network5.summary()


# In[47]:


#COMPILANDO
network5 = keras.utils.multi_gpu_model(network5,gpus=2)
network5.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])


# In[48]:


#TREINANDO

history5=network5.fit(train_images, train_labels,
                      batch_size=128,
                      epochs=140,
                      validation_split=0.2)


# In[49]:


#avaliando
Avalia(history5,network5,test_images,test_labels,1)


# # Rede 6: Convolucional com MaxPooling, Dropout e Weight Decay (norma L2) maior que na rede anterior

# In[51]:


#Definindo rede
network6 = models.Sequential()
network6.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',input_shape=input_shape))
network6.add(layers.Conv2D(64, (3, 3), activation='relu'))
network6.add(layers.Conv2D(64, (3, 3), activation='relu'))
network6.add(layers.MaxPooling2D(pool_size=(2, 2)))
network6.add(layers.Conv2D(64, (3, 3), activation='relu'))
network6.add(layers.MaxPooling2D(pool_size=(2, 2)))
network6.add(layers.Dropout(0.25))
network6.add(layers.Flatten())
network6.add(layers.Dense(128, activation='relu', activity_regularizer=keras.regularizers.l2(0.005)))
network6.add(layers.Dropout(0.5))
network6.add(layers.Dense(64, activation='relu', activity_regularizer=keras.regularizers.l2(0.005)))
network6.add(layers.Dropout(0.5))
network6.add(layers.Dense(10, activation='softmax'))

network6.summary()


# In[52]:


#COMPILANDO
network6 = keras.utils.multi_gpu_model(network6,gpus=2)
network6.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])


# In[53]:


#TREINANDO
history6=network6.fit(train_images, train_labels,
                      batch_size=128,
                      epochs=140,
                      validation_split=0.2)


# In[54]:


Avalia(history6,network6,test_images,test_labels,1)


# # Rede 7: MLP de 4 camadas

# In[75]:


#Observa-se que o MLP que foi feito (network1) teve underfitting, portanto, sera criado um MLP com mais capacidade

network7 = models.Sequential()
network7.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network7.add(layers.Dense(256, activation='relu'))
network7.add(layers.Dense(256, activation='relu'))
network7.add(layers.Dense(256, activation='relu'))
network7.add(layers.Dense(256, activation='relu'))
network7.add(layers.Dense(256, activation='relu'))
network7.add(layers.Dense(10, activation='softmax'))


# In[76]:


network7.summary()


# In[77]:


#COMPILANDO
network7 = keras.utils.multi_gpu_model(network7,gpus=2)
network7.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])


# In[78]:


#TREINANDO

history7 = network7.fit(train_images_mlp, train_labels, epochs=140, batch_size=128, validation_split=0.2)


# In[79]:


#avaliando
Avalia(history7,network7,test_images_mlp,test_labels,1)


# # Rede 8: MLP de 4 camadas com dropout

# In[80]:


#definindo rede
network8 = models.Sequential()
network8.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(256, activation='relu'))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(256, activation='relu'))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(256, activation='relu'))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(256, activation='relu'))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(256, activation='relu'))
network8.add(layers.Dropout(0.5))
network8.add(layers.Dense(10, activation='softmax'))

network8.summary()


# In[81]:


#paralelizando, compilando e treinando
network8 = keras.utils.multi_gpu_model(network8,gpus=2)
network8.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])
history8 = network8.fit(train_images_mlp, train_labels, epochs=140, batch_size=128, validation_split=0.2)


# In[82]:


#avaliando
Avalia(history8,network8,test_images_mlp,test_labels,1)


# # Rede 9: MLP de 4 camadas com weight decay (norma L2)

# In[84]:


#definindo rede

network9 = models.Sequential()
network9.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,),
                          activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(512, activation='relu', activity_regularizer=keras.regularizers.l2(0.0001)))
network9.add(layers.Dropout(0.5))
network9.add(layers.Dense(10, activation='softmax'))

network9.summary()


# In[85]:


#paralelizando, compilando e treinando
network9 = keras.utils.multi_gpu_model(network9,gpus=2)
network9.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])
history9 = network9.fit(train_images_mlp, train_labels, epochs=140, batch_size=128, validation_split=0.2)


# In[86]:


Avalia(history9,network9,test_images_mlp,test_labels,1)

