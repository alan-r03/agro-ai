import keras
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

def build_model():
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
