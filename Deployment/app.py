from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

# Load the trained model
model_path = "model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def preprocess_image(image):
    # Resize image to match model input size (224x224) and convert to numpy array
    image = image.resize((224, 224))
    image_array = np.array(image)
    # Normalize pixel values (if required)
    image_array = image_array / 255.0
    # Reshape array if necessary
    image_array = image_array.reshape(1, -1)  # Reshape according to model input shape
    return image_array

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle image upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']
    # Read the image file
    image = Image.open(file)
    # Preprocess the image
    image_array = preprocess_image(image)
    # Make predictions
    prediction = model.predict(image_array)
    # Convert the prediction to text
    gender = "Male" if prediction == 0 else "Female"
    # Return the predicted gender as JSON
    return jsonify({'gender': gender})

if __name__ == '__main__':
    app.run(debug=True)
