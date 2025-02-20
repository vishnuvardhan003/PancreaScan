import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Loading trained models
cnn_model = tf.keras.models.load_model('save.h5')  
random = pickle.load(open('pancreatic_random.pkl', 'rb'))  
naive = pickle.load(open('pancreatic_naive.pkl', 'rb'))  

image_labels = ['Normal', 'Pancreatic Tumor'] 
structured_labels = {
    1: 'Control (no pancreatic disease)',
    2: 'Benign (benign hepatobiliary disease)',
    3: 'PDAC (Pancreatic ductal adenocarcinoma)'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/criticalness')
def criticalness():
    return render_template('criticalness.html')

@app.route('/imageup')
def imageup():
    return render_template('imageup.html')

@app.route('/nearest-hospital')
def nearest_hospital():
    return render_template('nearh.html')

@app.route('/thingstoknow')
def thingstoknow():
    return render_template('thingstoknow.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    image = load_img(filepath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = cnn_model.predict(image)
    predicted_label = image_labels[np.argmax(prediction)]
    os.remove(filepath)

    return jsonify({'prediction': predicted_label})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Prepare the features and perform the prediction
        features = np.array([[float(data["patient_cohort"]),
                              float(data["sample_origin"]),
                              float(data["age"]),
                              float(data["sex"]),
                              float(data["stage"]),
                              float(data["benign_sample_diagnosis"]),
                              float(data["plasma_ca19_9"]),
                              float(data["creatinine"]),
                              float(data["lyve1"]),
                              float(data["reg1b"]),
                              float(data["tff1"]),
                              float(data["reg1a"])]])
        
        # Model selection and prediction
        model_choice = data["model"]
        
        if model_choice == "RandomForestClassifier":
            rf_pred = random.predict(features)[0]
            prediction = structured_labels[rf_pred]
            
        # elif model_choice == "naivebayes":
        #     nb_pred = naive.predict(features)[0]
        #     prediction = structured_labels[nb_pred]
            
        else:
            return jsonify({'error': 'Model not recognized'}), 400

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
