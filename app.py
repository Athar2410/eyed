from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from flask_cors import CORS   


app = Flask(__name__)
CORS(app)  # ✅ ADD THIS LINE TO ENABLE CORS

os.makedirs("temp", exist_ok=True)  # Temporary folder to store uploaded images

# Load your trained image-based CNN model
model = load_model('D:/pe2/ml-backend/ml-backend/model.h5')  # Update path if needed

# Define your class labels in the same order as your model's training
class_labels = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal", "Macular Degeneration"]

# Home route to check if API is running
@app.route('/')
def home():
    return "Image-based Eye Disease Prediction API is running!"

# Prediction route for image input
@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    try:
        img_file = request.files['image']
        img_path = os.path.join("temp", img_file.filename)
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

        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
