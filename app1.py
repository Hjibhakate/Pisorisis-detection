# ==================== Step 1: Imports ====================
from flask import Flask, render_template, request
import os
import numpy as np
from werkzeug.utils import secure_filename
from uuid import uuid4

# TensorFlow / Keras for image model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# ==================== Step 2: Flask Setup ====================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# ==================== Step 3: Directory Setup ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== Step 4: Load MobileNetV2 Model ====================
mobilenet_model_path = os.path.join(MODEL_DIR, "pisoriasis_mobilenetv2_final.h5")
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3),
    pooling=None,
)
for layer in base_model.layers:
    layer.trainable = False
base_model.load_weights(mobilenet_model_path, by_name=True, skip_mismatch=True)

inputs = layers.Input(shape=(224, 224, 3), name="input_layer_1")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
x = layers.Dropout(0.3, name="dropout")(x)
x = layers.Dense(128, activation="relu", name="dense")(x)
x = layers.Dropout(0.2, name="dropout_1")(x)
outputs = layers.Dense(1, activation="sigmoid", name="dense_1")(x)
mobilenet_model = Model(inputs=inputs, outputs=outputs, name="psoriasis_mobilenetv2")
mobilenet_model.load_weights(mobilenet_model_path, by_name=True, skip_mismatch=True)
mobilenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("MobileNetV2 loaded successfully!")

# ==================== Step 5: Lightweight Medical Report Setup ====================
print("Loading built-in medical report generator...")
print("Medical report generator ready!")




# ==================== Step 6: Model Parameter Info ====================
model_info = {
    "MobileNetV2": {
        "Accuracy": 94.5,
        "GFLOPs": 0.3,
        "Suitability": "Lightweight, real-time image classification"
    },
}

# ==================== Step 7: Helper Functions ====================

def predict_image(img_path):
    """Predict whether the skin image shows Psoriasis or Normal using MobileNetV2."""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get raw prediction probability
    prob = mobilenet_model.predict(img_array)[0][0]

    # Confidence calculation
    confidence = round(float(prob * 100), 2) if prob > 0.5 else round(float((1 - prob) * 100), 2)

    # Final class
    label = "Psoriasis" if prob > 0.5 else "Normal"

    return label, confidence

def generate_report(prediction):
    """Generate a lightweight explanatory report for the prediction result."""
    if prediction == "Psoriasis":
        return (
            "The uploaded skin image was classified as Psoriasis by the image model. "
            "Psoriasis is a chronic skin condition often associated with red, thick, "
            "and scaly patches. This result is only a screening prediction from the model "
            "and should not be treated as a final medical diagnosis. A dermatologist should "
            "review the symptoms, medical history, and physical examination before any treatment decision."
        )

    return (
        "The uploaded skin image was classified as Normal by the image model. "
        "This means the model did not detect psoriasis-like visual patterns strongly enough "
        "in the uploaded image. This prediction is only an automated screening result and "
        "does not replace professional medical evaluation. If symptoms such as itching, redness, "
        "or scaling continue, a dermatologist should still be consulted."
    )

# ==================== Step 8: Flask Routes ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 9a: Upload file
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filename = secure_filename(file.filename)
    if not filename:
        return "Invalid file name"

    unique_filename = f"{uuid4().hex}_{filename}"
    img_path = os.path.join(UPLOAD_DIR, unique_filename)
    file.save(img_path)

    # Step 9b: Image prediction + confidence
    prediction, confidence = predict_image(img_path)

    # Step 8c: Generate AI medical report
    report = generate_report(prediction)

    # Step 8d: Render result page with model parameters and confidence
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        report=report,
        image_path=f"static/uploads/{unique_filename}",
        model_info=model_info
    )

# ==================== Step 9: Run Flask App ====================
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
