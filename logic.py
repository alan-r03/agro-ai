from flask import render_template, session, request, redirect, url_for, jsonify
import keras.api
from app import app
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired
from sklearn.model_selection import train_test_split
from PIL import Image
import os, cv2, pandas as pd, numpy as np, time, keras, tensorflow as tf, shutil

def make_gradcam_heatmap(imgArr, model, pred_index=None):
    # Create model mapping input -> attention layer + output
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer("attention_layer").output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(imgArr)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_overlay_heatmap(imgName, imgPath, hmpArr, alpha=0.4):
    img = cv2.imread(imgPath)

    hmpName = f'heatmap_{imgName}'
    hmpArr = cv2.resize(hmpArr, (img.shape[1], img.shape[0]))
    hmpArr = np.uint8(255 * hmpArr)
    hmpColor = cv2.applyColorMap(hmpArr, cv2.COLORMAP_JET)
    hmpOverlay = cv2.addWeighted(img, 1 - alpha, hmpColor, alpha, 0)

    hmpPath = os.path.join(os.path.dirname(__file__), 'app', 'static', 'imgHeatmap', f'heatmap_{imgName}')
    os.makedirs(os.path.dirname(hmpPath), exist_ok=True)
    cv2.imwrite(hmpPath, hmpOverlay)

    return hmpName

def get_data():
    # step 1 - preprocess csv, get labels, randomize data
    path = os.path.join(os.path.dirname(__file__), 'app', 'static', 'imgAnnotations.csv')
    df = pd.read_csv(path, index_col=0)
    images = []
    labels = []
    prev = None

    for index, row in df.iterrows():        
        if index == prev:
            continue
        prev = index
        images.append(index)
        if all(row.iloc[:4] == 0):
            labels.append(0)
        else:
            labels.append(1)
    
    return images, labels

def create_model():

    # step 2 - create model, compile model, save bare model to static
    inputLayer = keras.Input(shape=(256, 256, 3), name='input_layer')
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputLayer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="attention_layer")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputLayer, outputs=output)
    print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train):
    batch_size = 10
    train_len = len(x_train)

    for i in range(0, train_len, batch_size):
        batch_imgs = []
        batch_labels = []

        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        for img_name in batch_x:
            img_path = os.path.join(os.path.dirname(__file__), 'app', 'static', 'imgHandheld', img_name)
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            batch_imgs.append(img_arr)

        batch_imgs = np.array(batch_imgs)
        batch_labels = np.array(batch_y).astype(np.float32).reshape(-1, 1)

        model.fit(batch_imgs, batch_labels, epochs=1, batch_size=len(batch_imgs), verbose=1)

    return batch_imgs, batch_labels