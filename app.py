from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "http://127.0.0.1:5500"}})
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temporary folder
os.makedirs("temp", exist_ok=True)

# Load model
try:
    model = load_model('D:/pe2/ml-backend/ml-backend/model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define class labels
class_labels = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal", "Macular Degeneration"]

@app.route('/')
def home():
    return "Image-based Eye Disease Prediction API is running!"

@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        logger.warning("No image file uploaded")
        return jsonify({'error': 'No image file uploaded'}), 400

    try:
        img_file = request.files['image']
        # Use a secure filename
        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join("temp", filename)
        img_file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(prediction))

        logger.info(f"Prediction: {predicted_label}, Confidence: {confidence * 100:.2f}%")

        # Clean up temporary file
        os.remove(img_path)

        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)