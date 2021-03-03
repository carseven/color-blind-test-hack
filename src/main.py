import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.python.keras.models as models
import json
import os

# Cargar modelo previamiente entrenado
pwd = os.getcwd()  # Ojo que es la ruta de este archivo, no la del proyecto
with open(pwd + '/model-data/model-config.json', 'r') as json_file:
    architecture = json.load(json_file)
    model = models.model_from_json(architecture)
    # model.load_weights(pwd + '/model-data/checkpoint')


def imshow(img):
    fig, ax = plt.subplots()  # solved by add this line
    plt.imshow(img, cmap='gray')
    return fig


def main():
    # Titulo de la página
    st.title("Color Blind Test Hack")
    st.subheader("Aplication develop by Carles Serra Vendrell")

    uploaded_file = st.file_uploader("Choose an image...", type="png")
    if uploaded_file is not None:
        # Todo: Pasar a procesarConvertir imagen en array
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8)
        # opencv_image = cv2.imdecode(file_bytes, 1)

        # Mostrar imagen subida
        st.image(
            uploaded_file,
            caption='Uploaded Image.',
            use_column_width=True)

        # Resize imagen
        # img = cv2.resize(opencv_image, (128, 128), 0, 0, cv2.INTER_AREA)

        # img = procesar_img(img)
        # st.pyplot(imshow(img),
        #           caption='Uploaded Image.',
        #           use_column_width=True)

        # # Todo los que sea procesado pasar a la función de procesar
        # processed_images = np.array(imutils.resize(img, height=28))
        # processed_images = np.array(processed_images)
        # processed_images = processed_images.reshape(1, 28, 28, 1)
        # processed_images = tf.cast(processed_images, tf.float32)
        # preds = np.argmax(new_model.predict(processed_images), axis=1)

        # st.write("")
        # st.write(f"El número es {preds[0]}")


if __name__ == '__main__':
    main()
