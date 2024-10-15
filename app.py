from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and class names
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'my_ayurvedic_plant_usingCNNmodel.h5')
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Aloevera', 'Amla', 'Bringaraja', 'Ginger', 'Hibiscus', 'Lemon', 'Mint', 'Neem', 'Tulsi', 'Ashoka']

# Resize target for images
IMAGE_RES = 224

# Prediction function
def predict_image(image_path):
    input_image = load_img(image_path, target_size=(IMAGE_RES, IMAGE_RES))
    input_image_array = img_to_array(input_image)
    input_image_array = input_image_array / 255.0  # Normalize the image
    input_image_array = np.expand_dims(input_image_array, axis=0)  # Add batch dimension

    predictions = model.predict(input_image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the file in the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Make prediction
        predicted_class = predict_image(filepath)
        
        # Render template with prediction and image
        return render_template('index.html', predicted_class=predicted_class, uploaded_image=file.filename)
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
