# Treinamento de uma Rede Neural Convolucional

# Imports
import os
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten



os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalização
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'amostras de treino')
print(x_test.shape[0], 'amostras de teste')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(name="convolution2d_1", filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(name="convolution2d_2", filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(name="maxpooling2d_1", pool_size=[2, 2]))
model.add(Dropout(name="dropout_1", rate=0.25))
model.add(Flatten())
model.add(Dense(name="dense_1", units=128, activation='relu'))
model.add(Dropout(name="dropout_2", rate=0.5))
model.add(Dense(name="dense_2", units=num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Treinamento
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Acurácia
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss de Teste:', score[0])
print('Acurácia de Teste:', score[1])


# Salvando do Modelo e Serializando com JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model/model.h5")
print("Modelo salvo em disco")
