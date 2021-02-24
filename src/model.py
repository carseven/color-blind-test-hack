import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import random
import json


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#  Las imagenes se convierten en tensores de 3 dimnsiones para poder ser
#  con las conv2d de keras.
train_images = train_images.reshape((60000, 28, 28, 1))

#  Se normalizan las imagenes en un factor 1/255 y se convierten en tipo float
train_images = train_images.astype('float32') / 255

#  Las imagenes se convierten en tensores de 3 dimnsiones para poder ser
#  con las conv2d de keras.
test_images = test_images.reshape((10000, 28, 28, 1))

#  Se normalizan las imagenes en un factor 1/255 y se convierten en tipo float
test_images = test_images.astype('float32') / 255

#  Se codifican las etiquetas como one-hot enconding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

"""### Aumentación de datos"""


#  Función propia, ruido gaussiano

def ruido(imagen):
    varianza = 0.1
    desviacion = varianza * random.random()
    ruido = np.random.normal(0, desviacion, imagen.shape)
    imagen += ruido
    np.clip(imagen, 0., 255.)
    return imagen


# Configuración del generador de imagenes.
datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             preprocessing_function=ruido)

#  Solo utilizamos aumentación en el conjunto de entrenamiento. Se indica al
#  al generador que imagenes tiene que procesar
datagen.fit(train_images)


#  Se indica que es un modelo secuencial
model = models.Sequential()

#  Se añaden las capas al modelo

#  Bloque 1 CNN
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True,
                        input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

#  Bloque 2 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Bloque 3 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

#  Bloque 4 FC
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

#  Se configura la función de perdidas y el algoritmo de apredizaje.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#  Visualización de los bloques y parametros del modelo implementado.
model.summary()

#  Se indica que datos alimentan al modelo en la fase de entrenamiento y en la
# de validación. En este caso los datos de entrenamiento viene generador tras
# procesar el conjunto de entrenamiento.
history = model.fit(datagen.flow(train_images, train_labels,
                                 batch_size=256),
                    steps_per_epoch=int(train_images.shape[0] / 256) + 1,
                    epochs=20,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

pwd = '/Users/carseven/dev/color-blind-test-hack/'

model.save_weights(pwd + 'src/model-data/mnist.tf', save_format='tf')

model_config = model.to_json()
with open(pwd + 'src/model-data/model-config.json',
          'w',
          encoding='utf-8') as f:
    json.dump(model_config, f, ensure_ascii=False, indent=4)
