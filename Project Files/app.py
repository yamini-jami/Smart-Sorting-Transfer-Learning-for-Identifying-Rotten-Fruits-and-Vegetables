from flask import Flask, request, jsonify,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np

app = Flask(__name__)

class_indices =  {'Apple__Healthy': 0, 'Apple__Rotten': 1, 'Banana__Healthy': 2, 'Banana__Rotten': 3, 'Bellpepper__Healthy': 4, 'Bellpepper__Rotten': 5, 'Carrot__Healthy': 6, 'Carrot__Rotten': 7, 'Cucumber__Healthy': 8, 'Cucumber__Rotten': 9, 'Grape__Healthy': 10, 'Grape__Rotten': 11, 'Guava__Healthy': 12, 'Guava__Rotten': 13, 'Jujube__Healthy': 14, 'Jujube__Rotten': 15, 'Mango__Healthy': 16, 'Mango__Rotten': 17, 'Orange__Healthy': 18, 'Orange__Rotten': 19, 'Pomegranate__Healthy': 20, 'Pomegranate__Rotten': 21, 'Potato__Healthy': 22, 'Potato__Rotten': 23, 'Strawberry__Healthy': 24, 'Strawberry__Rotten': 25, 'Tomato__Healthy': 26, 'Tomato__Rotten': 27}

class_index = {v: k for k, v in class_indices.items()}

# Load the pre-trained model
model = load_model('vgg16_detector.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        else:
            img_path = f"./static/uploads/{file.filename}"
            file.save(img_path)

            #preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 

            #predicting the class
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            # predicted_class_name = [key for key, value in class_indices.items() if value == predicted_class_index][0]
            predicted_class_name = class_index[predicted_class_index]
            label = predicted_class_name
            return render_template('index.html', label=label,img_path=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)