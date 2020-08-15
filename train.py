import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
import os
import cv2
import random
import sklearn.model_selection as model_selection
import datetime
from model import createModel


categories = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]

SIZE = 120


def getData():
    rawdata = []
    data = []
    dir = "./data/"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_data = cv2.resize(rawdata, (SIZE, SIZE))

                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data).reshape(-1, SIZE, SIZE, 1)
    img_data = img_data / 255.0
    img_labels = np.array(img_labels)

    return img_data, img_labels


# train_data, train_labels = Data("train")
# test_data,test_labels = Data("test")
data, labels = getData()
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.20)
print(len(train_data), " ", len(train_labels), len(test_data), " ", len(test_labels))

model = createModel(train_data)


checkpoint = keras.callbacks.ModelCheckpoint(filepath='./model/model.h5', save_best_only=True, monitor='val_loss',
                                             mode='min')

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"], )
#

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels),
                 )

model.save('./model/model.h5')
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Model Accuracy: ", test_acc, "Model Loss: ", test_loss)

# nimage = cv2.imread("26 (25).jpg.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(nimage,(SIZE,SIZE))
# image = image/255.0
# prediction = model.predict(np.array(image).reshape(-1,SIZE,SIZE,1))
# pclass = np.argmax(prediction)
# plt.imshow(image,cmap="gray")
# pValue = "Prediction: {0}".format(categories[int(pclass)])
# plt.title(pValue)
# realvalue = "Real Value {0}".format(categories[1])
# plt.figtext(0,0,realvalue)
# plt.show()


# print(train_data.shape)
# print(model.summary())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()