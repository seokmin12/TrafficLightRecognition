import cv2
import matplotlib.pyplot as plt
import numpy as np

lights = ['green', 'left_arrow', 'others_arrow', 'red', 'yellow']


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

model.load_weights('model/ckeckpointer.ckpt')

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("sample_data/image/i0008555.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        X = []
        if label == 'traffic light':
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            traffic_light = img[y:y + h, x:x + w]
            traffic_light = cv2.resize(traffic_light, (100, 60))
            X.append(traffic_light)

            x_data = np.array(X)
            score = model.predict(x_data)
            print(lights[np.argmax(score)])
            cv2.putText(img, lights[np.argmax(score)], (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()
