from flask_cors import CORS
app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("final_resnet.h5")
class_names = {
    0: 'FreshApple', 1: 'FreshBanana', 2: 'FreshGrape', 3: 'FreshGuava',
    4: 'FreshJujube', 5: 'FreshOrange', 6: 'FreshPomegranate', 7: 'FreshStrawberry',
    8: 'RottenApple', 9: 'RottenBanana', 10: 'RottenGrape', 11: 'RottenGuava',
    12: 'RottenJujube', 13: 'RottenOrange', 14: 'RottenPomegranate', 15: 'RottenStrawberry'
}

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    img = preprocess_image(file.read())
    pred = model.predict(img)[0]
    sorted_indices = np.argsort(pred)[::-1]
    top1, top2 = sorted_indices[:2]
    top1_label = class_names[top1]
    top2_label = class_names[top2]
    top1_conf = pred[top1] * 100
    top2_conf = pred[top2] * 100

    result = {}

    if top1_conf >= 80:
        result['message'] = 'Fresh' if 'Fresh' in top1_label else 'Rotten'
    elif ('Fresh' in top1_label and 'Rotten' in top2_label) or ('Rotten' in top1_label and 'Fresh' in top2_label):
        result['message'] = f"{top1_label} ({top1_conf:.1f}%) vs {top2_label} ({top2_conf:.1f}%)"
    elif 'Rotten' in top1_label:
        result['message'] = '⚠️ Do not eat this fruit.'
    elif 'Fresh' in top1_label and top1_conf < 80:
        result['message'] = '🍃 Seems fresh, but be cautious.'

    result['confidence'] = round(top1_conf, 2)
    result['fruitType'] = top1_label
    result['fresh'] = 'Fresh' in top1_label
    return jsonify(result)
