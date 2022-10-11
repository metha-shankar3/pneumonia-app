#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.utils import load_img,img_to_array
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/My_model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150)) #target_size must agree with what the trained model expects!!
     # Preprocessing the image
    img = img_to_array(img)
    print(img.shape)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    img = img.reshape(-1,150,150,1)
    preds = (model.predict(img) > 0.5).astype("int32")
    preds = preds.reshape(1,-1)[0][0]
    print(preds)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    classes = {'TRAIN':['Normal.So dont worry','Pneumonia Consult doctor']}
    
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        predicted_class = classes['TRAIN'][preds]
        print('I think that is {}.'.format(predicted_class.capitalize()))
        return str(predicted_class).capitalize()
        os.remove(file_path)#removes file from the server after prediction has been returned

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run(debug=True)

