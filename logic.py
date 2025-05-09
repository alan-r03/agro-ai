import numpy as np
import tensorflow as tf
import os
import cv2
import keras

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

def green_mask_crop(images):
    out = []

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    for image in images:
        if image is None or not isinstance(image, np.ndarray):
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # fallback to resizing the whole image
            resized = cv2.resize(image, (256, 256))
            out.append(resized)
            continue

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (256, 256))
        out.append(resized)

    return np.array(out)