from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# Model loading
model = tf.keras.models.load_model('fractured_tooth_detector.h5')

# Define image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
  img = load_img(image_path, target_size=target_size)
  img_array = img_to_array(img)
  img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

# Initialize Flask app
app = Flask(__name__)
#app = Flask(__name__, template_folder='templates')


# Route for homepage
@app.route("/")
def index():
  return render_template('index.html')  # Renders index.html template

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
  # Get uploaded image
  image_file = request.files['image']
  if not image_file:
    return jsonify({'message': 'No image uploaded'}), 400

  # Save image to temporary location
  image_path = f'temp_image.{image_file.filename.split(".")[1]}'
  image_file.save(image_path)

  # Preprocess image
  preprocessed_image = preprocess_image(image_path)

  # Make prediction
  predictions = model.predict(preprocessed_image)
  predicted_class = predictions[0] > 0.5

  # Prepare response
  if predicted_class:
    message = "Model Prediction: No Fracture Detected"
  else:
    message = "Model Prediction:  Fracture Detected"

  return jsonify({'message': message})

if __name__ == '__main__':
  app.run()  # Run the Flask app in debug mode
