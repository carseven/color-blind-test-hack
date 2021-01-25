import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

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
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

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

# Se grafican las primeras 9 muestras generadas por ImageDataGenerator
for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break

"""### Definir modelo entrenamiento"""


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

"""### Fase de entrenamiento"""

#  Se indica que datos alimentan al modelo en la fase de entrenamiento y en la
# de validación. En este caso los datos de entrenamiento viene generador tras
# procesar el conjunto de entrenamiento.
history = model.fit(datagen.flow(train_images, train_labels,
                                 batch_size=256),
                    steps_per_epoch=int(train_images.shape[0] / 256) + 1,
                    epochs=40,
                    validation_data=(test_images, test_labels))

"""### Fase de test"""

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

"""### Gráficos de la función de perdidas y de aciertos"""

#  Se obtiene los datos la función de perdidas calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
loss = history.history['loss']

#  Se obtiene los datos la función de perdidas calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
val_loss = history.history['val_loss']

#  Generación del vector de épocas
epochs = range(1, len(loss) + 1)

#  Generamos el grafico de la función de perdidas en entrenamiento y validación
plt.plot(epochs, loss, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Conjunto validación')

#  Se configura la apariencia del gráficos
plt.title('Curva de perdidas', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

#  Guardar en formato .png y mostrar el gráfico.
plt.savefig('loss.png')
plt.show()

#  Limpiar representación anterior.
plt.clf()


#  Se obtiene los datos la función de aciertos calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

#  Generamos el grafico de la función de perdidas en
#  entrenamiento y validación.
plt.plot(epochs, acc, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Conjunto validación')
plt.title('Curva de aciertos', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

#  Guardar en formato .png y mostrar el gráfico.
plt.savefig('val.png')
plt.show()

"""### Guardar modelo"""

model.save('mnist.h5')
