import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np

file = r"D:\\DATA SETS\\data"
categories = ["with_mask","without_mask"]

data = []
labels = []

for category in categories:
    path = os.path.join(file,category)
    for image in os.listdir(path):
        img_path = os.path.join(path,image)
        try:
            image = load_img(img_path,target_size = (224,224))
            image = img_to_array(image)
            image = image/255
            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"Error loading image{img_path}:{e}")

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=42)

model = Sequential([
    Conv2D(32,(3,3), activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3), activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2)
])

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs = 10, validation_data = (X_test,y_test))
model.evaluate(X_test,y_test)
model.save("mask_detector.h5")