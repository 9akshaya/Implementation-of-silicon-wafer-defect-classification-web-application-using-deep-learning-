from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 
import os
from os.path import join

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import layers, Input, models
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

datapath = join('data', 'wafer')


import warnings
warnings.filterwarnings("ignore") 



  
from PIL import Image

def create_model():
    input_shape = (26, 26, 3)
    input_tensor = Input(input_shape)

    conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
    conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
    conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)

    flat = layers.Flatten()(conv_3)

    dense_1 = layers.Dense(512, activation='relu')(flat)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    output_tensor = layers.Dense(9, activation='softmax')(dense_2)

    model = models.Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model 

app = Flask(__name__)

# Load the trained model
model_path = "C:\\Users\\ADMIN\\Desktop\\flask project\\model8.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Open and preprocess the image
            images = Image.open(uploaded_file) 
            image= images[:,:,1]
            x = np.array(image)
            x = x.reshape((-1, 26, 26, 1))
            new_x = np.zeros((len(x), 26, 26, 3))
            for w in range(len(x)):
                for i in range(26):
                    for j in range(26):
                        new_x[w, i, j, int(x[w, i, j])] = 1

            # Make predictions
            prediction = model.predict(new_x)

            # Decode predictions
            classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "random", "Scartch", "Near-Full", "None"]
            output = classes[np.argmax(prediction)]

            # Redirect to a new page with the prediction result
            return redirect(url_for('result', prediction=output))

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
