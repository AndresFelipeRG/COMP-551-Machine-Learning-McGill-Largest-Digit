

```python
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.utils import shuffle
import csv
```

    /Users/andresfeliperincongamboa/anaconda/envs/handwriting/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
from keras.layers import BatchNormalization
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python

from keras.layers import Dense, Dropout, Activation, Flatten
```


```python
import pandas as pd
```

## Image Preprocessing


```python
d1 = pd.read_csv("train_x.csv", sep = ",", header=None)
```


```python
d2 = pd.read_csv("test_x.csv", sep = ",", header=None)
```


```python
d3 = pd.read_csv("train_y.csv", sep = ",", header=None)
```


```python
def process(im, filtering):
    im2 = [1.0 if x <= filtering else 0.0 for x in np.array(im)]
    return im2
params = [230,240,250]
```


```python
datasets = [d1.apply(process,filtering = par, axis = 1) for par in params]
```


```python
dataset_test_real = [d2.apply(process,filtering = par, axis = 1) for par in params]
```


```python
for d in datasets:
    d["target"] = d3[0] 
```


```python
data = np.concatenate((datasets[0], datasets[1], datasets[2]), axis=0)
```


```python
data_test = np.concatenate((dataset_test_real[0], dataset_test_real[1], dataset_test_real[2]), axis=0)
```


```python
from random import shuffle
ids = [i for i in range(50000)]
shuffle(ids)
```


```python
train_ids = ids[10000:]
test_ids = ids[0:10000]
```


```python
a = (np.array(test_ids)).tolist()
b = (np.array(test_ids)+ 50000).tolist()
c = (np.array(test_ids)+ 2*50000).tolist()
d = np.concatenate([a,b,c]).tolist()
```


```python
a1 = (np.array(train_ids)).tolist()
b1 = (np.array(train_ids)+ 50000).tolist()
c1 = (np.array(train_ids)+ 2*50000).tolist()
d1 = np.concatenate([a1,b1,c1]).tolist()
```


```python
train_data = data[d1].astype('float32')
test_data = data[d].astype('float32')
```


```python
len(train_data)
```




    120000




```python
len(test_data)
```




    30000




```python
x_train = [train_data[x][0:4096] for x in range(len(train_data))]
y_train = [train_data[x][4096] for x in range(len(train_data))]
```


```python
x_test = [test_data[x][0:4096] for x in range(len(test_data))]
y_test = [test_data[x][4096] for x in range(len(test_data))]
```


```python
width_image = 64
height_image = 64
```


```python
x_test = np.array(x_test).reshape((len(x_test),1, width_image,height_image ))

x_train = np.array(x_train).reshape((len(x_train),1, width_image,height_image ))


```


```python
x_test_real = np.array(data_test).reshape((len(data_test),1, width_image,height_image))


```


```python
x_test_real = x_test_real.astype('float32')

```


```python
y_test2 = np.zeros((len(y_test),10), dtype = "float32")
y_train2 = np.zeros((len(y_train),10), dtype = "float32")
for x in range(len(y_train)):
    y_train2[x][int(y_train[x])] = 1
for x in range(len(y_test)):
    y_test2 [x][int(y_test[x])] = 1
```


```python
len(x_test_real)
```




    30000




```python
x_train.shape[1:]
```




    (1, 64, 64)



# Creating first CNN model


```python
# Early stopping using Keras
from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')


model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
```


```python
model.add(Activation('relu'))
```


```python
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
```


```python
model.add(MaxPooling2D(pool_size=(2, 2)))

```


```python
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
```


```python
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
```


```python
model.add(MaxPooling2D(pool_size=(2, 2)))

```


```python
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```


```python
opt = keras.optimizers.Adam(lr=0.001)
```


```python
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```


```python
from keras.preprocessing.image import ImageDataGenerator
```


```python
#Data generation.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 45)
        width_shift_range=0.00,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.00,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
```


```python
datagen.fit(x_train)
```


```python
steps_per_epoch= len(x_train)/batch_size
```


```python
steps_per_epoch
```




    3843




```python
x_test.shape
```




    (27000, 1, 64, 64)




```python
model.fit_generator(datagen.flow(x_train[80000:], y_train2[80000:],batch_size=batch_size),steps_per_epoch =steps_per_epoch/3,epochs=30,validation_data=(x_test, y_test2),workers=4, callbacks = [early])
```


```python

preds = model.predict(x_test_real[20000:30000], verbose=1)
```

    10000/10000 [==============================] - 387s 39ms/step



```python
preds = pd.DataFrame(np.argmax(preds, axis=1).astype('float32'))
preds.to_csv("predictionsmodel1.csv",sep = ',')
```


```python
# Save the model
model.save('DIGITS_trained_model.h5')
```


```python
model.save_weights('DIGITS_trained_modelw.h5')
```

# Creating second CNN model


```python
#Data generation.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 10)
        width_shift_range=0.00,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.00,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
```


```python
model2 = Sequential()
model2.add(Convolution2D(32, (3, 3), padding = 'same', input_shape=(x_train.shape[1:])))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Convolution2D(32, (3, 3)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Convolution2D(64, (3, 3)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Convolution2D(64, (3, 3)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Convolution2D(64, (3, 3)))
```


```python
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Convolution2D(64, (3, 3)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
```


```python
model2.add(Dropout(0.3))
model2.add(Flatten())
model2.add(Dense(512))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.3))
model2.add(Dense(256))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.45))
model2.add(Dense(num_classes, activation='softmax'))
opt3 = keras.optimizers.Adam(lr=0.005)
model2.compile(loss='categorical_crossentropy',
              optimizer=opt3,
              metrics=['accuracy'])
```


```python
model2.fit_generator(datagen.flow(x_train[80000:], y_train2[80000:],batch_size=batch_size),steps_per_epoch =steps_per_epoch/3,epochs=30,validation_data=(x_test[20000:], y_test2[20000:]),workers=4, callbacks = [early])
```


```python

model2.save('DIGITS_trained_model2-v1.h5')
model2.save_weights('DIGITS_trained_modelw2-v1.h5') #score ?
```


```python
preds3 = model2.predict(x_test_real[20000:30000], verbose=1)
preds3 = pd.DataFrame(np.argmax(preds3, axis=1).astype('float32'))
preds3.to_csv("predictionsModel2.csv",sep = ',')

```

# Creating the third CNN Model


```python
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 45)
        width_shift_range=0.00,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.00,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)##don'T rotate images during training
```


```python
model3 = Sequential()

model3.add(Convolution2D(32, (3, 3), input_shape=(x_train.shape[1:])))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Convolution2D(32, (3, 3)))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Convolution2D(64, (3, 3)))
```


```python


model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Convolution2D(64, (3, 3)))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Convolution2D(128, (3, 3)))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Convolution2D(128, (3, 3)))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))

```


```python
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(512))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Dropout(0.3))
model3.add(Dense(256))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Dropout(0.3))
model3.add(Dense(200))

model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(Dropout(0.45))
model3.add(Dense(num_classes, activation='softmax'))
opt4 = keras.optimizers.Adam(lr=0.01)
model4.compile(loss='categorical_crossentropy',
              optimizer=opt4,
              metrics=['accuracy'])

```


```python

```


```python
# Early stopping using Keras
from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

```


```python
model3.fit_generator(datagen.flow(x_train[80000:], y_train2[80000:],batch_size=batch_size),steps_per_epoch =steps_per_epoch/3,epochs=30,validation_data=(x_test[20000:], y_test2[20000:]),workers=4, callbacks = [early])
```


```python
preds3 = model3.predict(x_test_real[20000:30000], verbose=1)
preds3 = pd.DataFrame(np.argmax(preds3, axis=1).astype('float32'))
preds3.to_csv("predictionsModel3.csv",sep = ',') 
```


```python
preds3 = model3.predict(x_test[20000:30000], verbose=1)
preds3 = pd.DataFrame(np.argmax(preds3, axis=1).astype('float32'))
preds4_compare = pd.DataFrame(np.argmax(y_test2[10000:20000],axis = 1).astype('float32'))
```


```python
def plot_confusion_matrix(cls_pred, cls_true):
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()
```


```python
#Plot confusion matrix
plot_confusion_matrix(preds3, preds4_compare)
```
