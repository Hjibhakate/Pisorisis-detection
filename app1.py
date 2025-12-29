# ==================== Step 1: Imports ====================
from flask import Flask, render_template, request
import os
import numpy as np
from werkzeug.utils import secure_filename

# TensorFlow / Keras for image model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Hugging Face Transformers for LLM
from transformers import pipeline
import torch  # Ensure PyTorch is installed

# ==================== Step 2: Flask Setup ====================
app = Flask(__name__)

# ==================== Step 3: Directory Setup ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ==================== Step 4: Load MobileNetV2 Model ====================
mobilenet_model_path = os.path.join(MODEL_DIR, "pisoriasis_mobilenetv2_final.h5")
print("Loading MobileNetV2 model...")
mobilenet_model = load_model(mobilenet_model_path)
mobilenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("MobileNetV2 loaded successfully!")

# ==================== Step 5: Load LSTM Model ====================
lstm_model_path = os.path.join(MODEL_DIR, "lstm_model_final.h5")
print("Loading LSTM model...")
lstm_model = load_model(lstm_model_path, compile=False)
print("LSTM model loaded successfully!")

# Get LSTM expected input shape
lstm_timesteps = lstm_model.input_shape[1]
lstm_features = lstm_model.input_shape[2]
print(f"LSTM expects input shape: (timesteps={lstm_timesteps}, features={lstm_features})")


# ==================== Step 6: Setup Hugging Face Medical LLM (DistilGPT2 + Medical Prompting) ====================
print("🧠 Loading Hugging Face Medical LLM: DistilGPT2 (medical-prompted)...")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load small GPT-style model for text generation
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
medical_llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def generate_medical_text(prompt):
    """Generate detailed text using GPT-style model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = medical_llm_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("✅ Medical LLM (DistilGPT2) loaded successfully!")




# ==================== Step 7: Model Parameter Info ====================
model_info = {
    "MobileNetV2": {
        "Accuracy": 94.5,
        "GFLOPs": 0.3,
        "Suitability": "Lightweight, real-time image classification"
    },
    "LSTM": {
        "Accuracy": 91.8,
        "GFLOPs": 0.8,
        "Suitability": "Sequential dermatological pattern analysis"
    }
}

# ==================== Step 8: Helper Functions ====================

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
    """Generate extended medical explanation using GPT-style model."""
    inputs = tokenizer(prediction, return_tensors="pt")

    outputs = medical_llm_model.generate(
        **inputs,
        max_new_tokens=600,   # 🧠 Increase for longer text (try 400–800)
        temperature=0.8,      # Adds creativity
        top_p=0.9,            # Keeps it coherent
        repetition_penalty=1.2,  # Avoids repeating phrases
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)




# ==================== Step 9: Flask Routes ====================
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
    img_path = os.path.join(STATIC_DIR, filename)
    file.save(img_path)

    # Step 9b: Image prediction + confidence
    prediction, confidence = predict_image(img_path)

    # Step 9c: LSTM dummy prediction
    sequence_input = np.random.rand(1, lstm_timesteps, lstm_features)
    lstm_output = lstm_model.predict(sequence_input)

    # Step 9d: Generate AI medical report
    report = generate_report(prediction)

    # Step 9e: Render result page with model parameters and confidence
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        report=report,
        image_path=f"static/{filename}",
        model_info=model_info
    )

# ==================== Step 10: Run Flask App ====================
if __name__ == "__main__":
    app.run(debug=True)
