import json
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint


def get_cropped_image():
    X = []
    Y = []
    for image in tqdm(glob('sample_data/image/*.jpg')):
        try:
            image_name = image.split('/')[2].split('.')[0]

            with open('sample_data/json/' + image_name + '.json') as f:
                file = json.load(f)

                for annotation in file['annotation']:
                    if annotation['class'] == 'traffic_light':
                        point = annotation['box']
                        light = {v: k for k, v in annotation['attribute'][0].items()}
                        Y.append(light.get('on'))

                        img = cv2.imread('sample_data/image/' + image_name + '.jpg')
                        pointed_img = img[point[1]:point[3], point[0]:point[2]]
                        pointed_img = cv2.resize(pointed_img, (100, 60))
                        X.append(pointed_img)
                    else:
                        continue

        except cv2.error as e:
            print(e)
            continue

    x_data = np.array(X)
    y_data = np.array(Y)

    # x_data = x_data.reshape(len(x_data), 20, 20, 1)
    y_data = pd.get_dummies(y_data)
    return x_data, y_data


x_data, y_data = get_cropped_image()


def create_model():
    from tensorflow.keras import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dropout

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(100, 60, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        Dropout(0.25),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        Dropout(0.25),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])

    return model


model = create_model()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print(x_data.shape, y_data.shape, x_train.shape, y_train.shape)

# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('model', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

history = model.fit(x_train, y_train, epochs=250, validation_split=0.4, callbacks=[checkpoint])

model.load_weights(filename)
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
