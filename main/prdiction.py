import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from PIL import Image

# Configure upload folder
UPLOAD_FOLDER = os.path.join(settings.MEDIA_ROOT, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the models
models = {
    "Alzheimers": {
        "model": load_model("/home/user/Desktop/Disease Prediction/model/dementia_classification_model.h5"),
        "class_labels": ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia'],
        "input_shape": (128, 128)
    },
    "Brain_tumor": {
        "model": load_model("/home/user/Desktop/Disease Prediction/model/BrainTumor.h5"),
        "class_labels": ['glioma', 'meningioma', 'notumor', 'pituitary'],
        "input_shape": (224, 224)
    },
    "Diabetic": {
        "model": load_model("/home/user/Desktop/Disease Prediction/model/Diabetic(1).h5"),
        "class_labels": ['DR', 'No_DR'],
        "input_shape": (224, 224)
    },
    "Kidney": {
        "model": load_model("/home/user/Desktop/Disease Prediction/model/KidneyCTscan(1).h5"),
        "class_labels": ['Cyst', 'Normal', 'Stone', 'Tumor'],
        "input_shape": (224, 224)
    },
    "Respiratory": {
        "model": load_model("/home/user/Desktop/Disease Prediction/model/Respiratory.h5"),
        "class_labels": ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia'],
        "input_shape": (128, 128)
    }
}

# Preprocess the image for the selected model
def preprocess_image(image_path, model_key):
    model_info = models.get(model_key)
    input_shape = model_info["input_shape"]

    if model_key == "Alzheimers":
        img = Image.open(image_path)
        img = img.resize(input_shape)
        img = np.array(img).reshape(1, input_shape[0], input_shape[1], 3)
    else:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, size=input_shape)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
    return img

# Home Page
def index(request):
    return render(request, 'index.html')

# Prediction Logic
def predict(request):
    if request.method == 'POST':
        try:
            model_key = request.POST.get('model_key')
            imagefile = request.FILES['imagefile']
            image_path = os.path.join(UPLOAD_FOLDER, imagefile.name)
            
            # Save file
            with default_storage.open(image_path, 'wb+') as destination:
                for chunk in imagefile.chunks():
                    destination.write(chunk)

            # Preprocess and predict
            img_array = preprocess_image(image_path, model_key)
            model = models[model_key]["model"]
            class_labels = models[model_key]["class_labels"]

            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100
            prediction = f"{confidence:.2f}% Confidence: {class_labels[class_index]}"

            image_url = f"{settings.MEDIA_URL}uploads/{imagefile.name}"
            return render(request, 'index.html', {'prediction': prediction, 'image_url': image_url})

        except Exception as e:
            return render(request, 'index.html', {'error': str(e)})

    return render(request, 'index.html')


# pip install django pillow tensorflow numpy
