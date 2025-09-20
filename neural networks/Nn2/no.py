import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from ultralytics import YOLO

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------
# TrashNet Classes
# -------------------------
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# -------------------------
# Load Models
# -------------------------
print("Loading TensorFlow/Keras models...")

autoencoder = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/denoising_autoencoder"
)

custom_objects = {'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy}
multimodal_model = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/multimodal_resnet_text",
    custom_objects=custom_objects
)

cnn_model = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/cnn_model.h5"
)

resnet_model = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/resnet_model_fixed.h5"
)

generator_model = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/generator_model.h5"
)
discriminator_model = tf.keras.models.load_model(
    r"C:/Users/Begad/Documents/training/final project/discriminator_model.h5"
)

print("✅ TensorFlow/Keras models loaded!")

# YOLO chunk models
print("Loading YOLO chunk models...")
yolo_chunks = [
    YOLO(r"C:/Users/Begad/Documents/training/final project/chunk/chunk1/weights/best.pt"),
    YOLO(r"C:/Users/Begad/Documents/training/final project/chunk/chunk2/weights/best.pt"),
    YOLO(r"C:/Users/Begad/Documents/training/final project/chunk/chunk3/weights/best.pt")
]
print("✅ YOLO chunk models loaded!")

# -------------------------
# Helper Functions
# -------------------------
def preprocess_image(img_path, model_name=None):
    if model_name in ["cnn_model", "autoencoder"]:
        target_size = (128, 128)
    else:
        target_size = (224, 224)
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_yolo(image_path):
    all_labels = []
    for model in yolo_chunks:
        results = model.predict(source=image_path)
        for r in results:
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                labels = [r.names[int(cls)] for cls in r.boxes.cls]
                all_labels.extend(labels)
    return all_labels

def generate_gan_image(label):
    label_index = class_names.index(label)
    latent_dim = 100
    noise = np.random.normal(0, 1, (1, latent_dim))
    label_array = np.array([[label_index]])
    generated_image = generator_model.predict([noise, label_array])
    generated_image = ((generated_image[0] + 1) * 127.5).astype(np.uint8)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gan_{label}.png")
    Image.fromarray(generated_image).save(save_path)
    return f"gan_{label}.png"

def get_prediction(file_path, model_name, text_input_value=None, gan_label=None):
    filename = None

    if model_name == "cnn_model":
        img = preprocess_image(file_path, model_name)
        preds = cnn_model.predict(img)
        pred_index = int(np.argmax(preds, axis=1)[0])
        prediction = class_names[pred_index]

    elif model_name == "resnet_model":
        img = preprocess_image(file_path, model_name)
        preds = resnet_model.predict(img)
        pred_index = int(np.argmax(preds, axis=1)[0])
        prediction = class_names[pred_index]

    elif model_name == "autoencoder":
        img = preprocess_image(file_path, model_name)
        preds = autoencoder.predict(img)
        if preds.ndim > 1 and preds.shape[1] == len(class_names):
            pred_index = int(np.argmax(preds, axis=1)[0])
            prediction = class_names[pred_index]
        else:
            prediction = "Autoencoder output (reconstructed image)"

    elif model_name == "multimodal_model":
        img_array = preprocess_image(file_path, model_name)
        text_input_value = text_input_value if text_input_value else ""
        text_array = np.array([text_input_value], dtype=object)
        preds = multimodal_model.predict([img_array, text_array])
        pred_index = int(np.argmax(preds, axis=1)[0])
        prediction = class_names[pred_index]

    elif model_name == "yolo":
        labels = predict_yolo(file_path)
        prediction = ", ".join(labels) if labels else "No objects detected"

    elif model_name == "gan":
        if gan_label in class_names:
            filename = generate_gan_image(gan_label)
            prediction = f"Generated GAN image for {gan_label}"
        else:
            prediction = "Invalid GAN label"

    else:
        prediction = "Unknown model"

    return prediction, filename

# -------------------------
# Routes
# -------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_file():
    selected_model = request.form.get('model')
    gan_label = request.form.get('gan_label')
    text_input_value = request.form.get('text_input')

    file_path = None
    filename = None

    if selected_model in ["cnn_model", "resnet_model", "autoencoder", "multimodal_model", "yolo"]:
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    prediction, gen_filename = get_prediction(file_path, selected_model, text_input_value, gan_label)
    return render_template('index.html', prediction=prediction, filename=filename or gen_filename)

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
