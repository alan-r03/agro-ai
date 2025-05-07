# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
from flask import render_template, session, request, redirect, url_for, jsonify
import keras.api
from app import app
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired
from sklearn.model_selection import train_test_split
from PIL import Image
from .builder import build_model
import os, cv2, pandas as pd, numpy as np, time, keras, tensorflow as tf, shutil

class LabelForm(FlaskForm):
    choice = RadioField(u'Label', choices=[(0, u'Healthy'), (1, u'Unhealthy')], validators = [DataRequired(message='Cannot be empty')])
    submit = SubmitField('Add Label')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'static', 'model.weights.keras')
tf.config.run_functions_eagerly(True)

@app.route("/", methods=['GET'])
@app.route("/index.html",methods=['GET'])
def home():
    """
    Operates the root (/) and index(index.html) web pages.
    """
    # wipe cookies
    session.clear()

    # step 1 - preprocess csv, get labels, randomize data
    path = os.path.join(os.path.dirname(__file__), 'static', 'imgAnnotations.csv')
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

    session['x_train'], session['x_test'], session['y_train'], session['y_test'] = train_test_split(images, labels, test_size=0.1, random_state=int(time.time()))

    # step 2 - create model, compile model, save bare model to static
    model = build_model()
    model.save(MODEL_PATH)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/label.html", methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    Initializes session data if needed and handles labeling logic.
    """
    # step 1 - verify user properly initialized app (thru home)
    if 'x_train' not in session or 'y_train' not in session:
        return redirect(url_for('home'))

    form = LabelForm()

    # step 2 - clear cookies necessary for labeling
    if 'form_images' not in session or 'form_labels' not in session:
        session['form_images'] = session['x_train'][-10:]
        session['form_labels'] = session['y_train'][-10:]
        session['x_train'] = session['x_train'][:-10]
        session['y_train'] = session['y_train'][:-10]
        if 'x_user' not in session or 'y_user' not in session:
            session['x_user'] = []
            session['y_user'] = []
        session['counter'] = 0
    elif len(session['form_images']) <= 0 or len(session['form_labels']) <= 0:
        return redirect(url_for('final'))

    # step 3 - collect user inputs through form handling
    if form.validate_on_submit():
        # store user's label for current image
        curr_idx = session['counter']
        session['form_labels'][curr_idx] = (form.choice.data)
        session['counter'] += 1

        # after 10 labels, clear cookies, store inputs in x_user and y_user
        if session['counter'] >= len(session['form_images']):
            session['x_user'] += (session.pop('form_images', None))
            session['y_user'] += (session.pop('form_labels', None))
            session.pop('counter', None)

            return redirect(url_for('intermediate'))

        return redirect(url_for('label'))

    # update counter and current image for display
    curr_idx = session['counter']
    curr_img = session['form_images'][curr_idx]
    img_path = url_for('static', filename=f'imgHandheld/{curr_img}')

    return render_template('label.html', form=form, img_path=img_path, curr_idx=curr_idx+1)

@app.route("/intermediate.html",methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    # step 1 - preprocess data
    x_user = session['x_user']
    y_user = session['y_user']
    x_user_maps = []

    x_test = session['x_test']
    y_test = session['y_test']
    x_test_maps = []
    
    for imgName in x_user:
        imgPath = os.path.join(os.path.dirname(__file__), 'static', 'imgHandheld', imgName)

        # Open image, convert to RGB, resize to 256x256
        img = Image.open(imgPath).convert('RGB').resize((256, 256))

        # Convert to NumPy array and normalize (0-255 -> 0-1)
        imgArr = np.array(img, dtype=np.float32) / 255.0
        x_user_maps.append(imgArr)

    x_user = np.array(x_user_maps)
    y_user = np.array(y_user).astype(np.float32).reshape(-1, 1)

    for imgName in x_test:
        imgPath = os.path.join(os.path.dirname(__file__), 'static', 'imgHandheld', imgName)

        # Open image, convert to RGB, resize to 256x256
        img = Image.open(imgPath).convert('RGB').resize((256, 256))

        # Convert to NumPy array and normalize (0-255 -> 0-1)
        imgArr = np.array(img, dtype=np.float32) / 255.0
        x_test_maps.append(imgArr)

    test_x = np.array(x_test_maps)
    test_y = np.array(y_test).astype(np.int32).reshape(-1, 1)

    # step 2 - load, compile, and feed model
    model = keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(x_user, y_user, epochs=1, batch_size=len(x_user), verbose=1)

    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    loss = f'{loss * 100:.2f}'
    accuracy = f'{accuracy * 100:.2f}'

    model.save(MODEL_PATH)

    # step 3 - sort user inputs to display on intermediate
    userInput = list(zip(session['x_user'][-10:], session['y_user'][-10:]))
    userH, userU = [], []

    for img, label in userInput:
        if label in '0':
            userH.append(url_for('static', filename=f'imgHandheld/{img}'))
        elif label in '1':
            userU.append(url_for('static', filename=f'imgHandheld/{img}'))

    print(userH, userU)

    return render_template('intermediate.html', accuracy=accuracy, loss=loss, userH=userH, lenH=len(userH), userU=userU, lenU=len(userU))

@app.route("/final.html", methods=["GET"])
def final():
    # Load model
    model = keras.models.load_model(MODEL_PATH)

    # Load train image names and labels
    x_train = session.get('x_train', [])
    y_train = session.get('y_train', [])

    # Process images and train model
    batch_size = 10
    train_len = len(x_train)

    for i in range(0, train_len, batch_size):
        batch_imgs = []
        batch_labels = []

        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        for img_name in batch_x:
            img_path = os.path.join(os.path.dirname(__file__), 'static', 'imgHandheld', img_name)
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            batch_imgs.append(img_arr)

        batch_imgs = np.array(batch_imgs)
        batch_labels = np.array(batch_y).astype(np.float32).reshape(-1, 1)

        model.fit(batch_imgs, batch_labels, epochs=1, batch_size=len(batch_imgs), verbose=1)

    model.save(MODEL_PATH)

    # Load test image names and labels
    x_test = session['x_test']
    y_test = session['y_test']

    test_imgs = []
    for imgName in x_test:
        imgPath = os.path.join(os.path.dirname(__file__), 'static', 'imgHandheld', imgName)
        img = Image.open(imgPath).convert('RGB').resize((256, 256))
        imgArr = np.array(img, dtype=np.float32) / 255.0
        test_imgs.append(imgArr)

    x_test_arr = np.array(test_imgs)
    y_test_arr = np.array(y_test).astype(np.int32).reshape(-1, 1)

    # Predict probabilities
    probs = model.predict(x_test_arr).flatten()
    preds = (probs > 0.5).astype(int)

    # Separate image URLs and confidence per predicted label
    modelH, modelU = [], []
    probArrH, probArrU = [], []

    for img_name, pred, prob in zip(x_test, preds, probs):
        if pred == 0:
            modelH.append(url_for('static', filename=f'imgHandheld/{img_name}'))
            probArrH.append(f'{(1 - prob) * 100:.2f}%')
        else:
            modelU.append(url_for('static', filename=f'imgHandheld/{img_name}'))
            probArrU.append(f'{prob * 100:.2f}%')

    # Calculate percentages
    total = len(x_test)
    lenMH = len(modelH)
    lenMU = len(modelU)
    percentH = f'{(lenMH / total) * 100:.2f}%' if total else '0%'
    percentU = f'{(lenMU / total) * 100:.2f}%' if total else '0%'

    # Pull user-labeled data
    user_data = list(zip(session['x_user'][-10:], session['y_user'][-10:]))
    userH = [url_for('static', filename=f'imgHandheld/{img}') for img, label in user_data if str(label) == '0']
    userU = [url_for('static', filename=f'imgHandheld/{img}') for img, label in user_data if str(label) == '1']
    lenUH = len(userH)
    lenUU = len(userU)

    # Evaluate to get confidence/accuracy
    loss, accuracy = model.evaluate(x_test_arr, y_test_arr, verbose=0)
    confidence = f'{accuracy * 100:.2f}%'

    return render_template('final.html', confidence=confidence, userH=userH, userU=userU, lenUH=lenUH,
                           lenUU=lenUU, modelH=modelH, modelU=modelU, lenMH=lenMH, lenMU=lenMU,
                           percentH=percentH, percentU=percentU, probArrH=probArrH, probArrU=probArrU)

@app.route('/gradcam', methods=['POST'])
def gradcam():
    imgName = request.json.get('image_path').split('/')[-1]
    imgPath = os.path.join(os.path.dirname(__file__), 'static', 'imgHandheld', imgName)

    # Preprocess image
    img = Image.open(imgPath).convert('RGB').resize((256, 256))
    imgArr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Load keras models
    model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'static', 'model.weights.keras'))

    # Generate heatmap
    hmpArr = make_gradcam_heatmap(imgArr, model)
    hmpName = save_and_overlay_heatmap(imgName, imgPath, hmpArr)

    return jsonify({'gradcam_url': url_for('static', filename=f'imgHeatmap/{hmpName}')})


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

    hmpPath = os.path.join(os.path.dirname(__file__), 'static', 'imgHeatmap', f'heatmap_{imgName}')
    os.makedirs(os.path.dirname(hmpPath), exist_ok=True)
    cv2.imwrite(hmpPath, hmpOverlay)

    return hmpName