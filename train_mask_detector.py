import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

datasetFolder = "/content/Face-Mask-Detection/dataset"
categories = ["with_mask", "without_mask"]

data = []
labels = []

for category in categories:
    path = os.path.join(datasetFolder, category)
    for filename in os.listdir(path):
        imgPath = os.path.join(path, filename)
        img = load_img(imgPath, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)

        data.append(img)
        labels.append(category)

# print(len(labels))
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# print(len(labels))
# labels = to_categorical(img)
# print(len(labels))

print(len(data))
data = np.array(data)
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data,
                                                labels,
                                                test_size = 0.2,
                                                stratify=labels,
                                                random_state=42)

aug = ImageDataGenerator(
    rotation_range = 20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

baseModel = MobileNetV2(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation='sigmoid')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
  layer.trainable = False

# print(model.summary())

lr = 1e-4
epoch = 20
batchSize = 32

opt = Adam(learning_rate=lr, decay=lr/epoch)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(aug.flow(trainX, trainY, batch_size = batchSize),
                    steps_per_epoch = len(trainX) // batchSize,
                    validation_data = (testX, testY),
                    validation_steps = len(testX) // batchSize,
                    epochs = epoch)

pred = model.predict(testX, batch_size=batchSize)
# print(pred)

pred = np.argmax(pred, axis=1)

print(classification_report(testY, pred,
                            target_names = lb.classes_))

model.save("mask_detector.model", save_format="h5")

N = epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")