from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
svm_model = joblib.load("model/svm_model.pkl")
resnet_model = load_model("model/resnet_feature_model.h5")

def preprocess_img(file):
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    img = preprocess_img(file)
    features = resnet_model.predict(img)
    prediction = svm_model.predict(features)
    result = "Parkinson" if prediction[0] == 1 else "Normal"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)