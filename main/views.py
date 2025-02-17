from django.shortcuts import render,redirect
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login
from .models import *
from .forms import *
import os
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from PIL import Image
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.utils.timezone import now
from datetime import timedelta
from django.core.paginator import Paginator


class LoginView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        log_form=LogForm(data=request.POST)
        if log_form.is_valid():  
            us=log_form.cleaned_data.get('username')
            ps=log_form.cleaned_data.get('password')
            user=authenticate(request,username=us,password=ps)
            if user: 
                login(request,user)
                return redirect('main')
            else:
                return render(request,'login.html',{"form":log_form})
        else:
            return render(request,'login.html',{"form":log_form}) 
        

class RegView(CreateView):
     form_class=UserForm
     template_name="register.html"
     model=CustomUser
     success_url=reverse_lazy("login")  


class MainPage(TemplateView):
    template_name = 'home.html'
    

class Prediction(TemplateView):
    template_name = 'prediction.html'
    

class HistoryView(TemplateView):
    template_name = 'history.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        user = self.request.user
        query_params = self.request.GET

        # Get initial queryset
        history_qs = History.objects.filter(user=user)

        # Filtering by date range
        date_range = query_params.get('date_range', 'all')
        if date_range == "last_7_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=7))
        elif date_range == "last_30_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=30))
        elif date_range == "last_3_months":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=90))

        # Filtering by model type
        model_type = query_params.get('model_type', '')
        if model_type and model_type != "All Models":
            history_qs = history_qs.filter(model_key=model_type)

        # Search by user input
        search_query = query_params.get('search', '')
        if search_query:
            history_qs = history_qs.filter(result__icontains=search_query)

        # Pagination
        paginator = Paginator(history_qs, 10)  # 10 items per page
        page_number = query_params.get('page', 1)
        history_page = paginator.get_page(page_number)

        context['history'] = history_page
        context['page_obj'] = history_page
        return context
    

def custom_logout(request):
    auth_logout(request)
    return redirect('login')



models = {
    "Alzheimers": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/dementia_classification_model.h5")),
        "class_labels": ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia'],
        "input_shape": (128, 128)
    },
    "Brain_tumor": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/BrainTumor.h5")),
        "class_labels": ['glioma', 'meningioma', 'notumor', 'pituitary'],
        "input_shape": (224, 224)
    },
    "Diabetic": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/Diabetic.h5")),
        "class_labels": ['DR', 'No_DR'],
        "input_shape": (224, 224)
    },
    "Kidney": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/KidneyCTscan.h5")),
        "class_labels": ['Cyst', 'Normal', 'Stone', 'Tumor'],
        "input_shape": (224, 224)
    },
    "Respiratory": {
        "model": load_model(os.path.join(settings.BASE_DIR, "models/Respiratory.h5")),
        "class_labels": ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia'],
        "input_shape": (128, 128)
    }
}

# Preprocess the image for the selected model

def preprocess_image(image_path, model_key):
    model_info = models.get(model_key)
    input_shape = model_info["input_shape"]

    # Open the image
    img = Image.open(image_path)

    # Convert to RGB (force 3 channels)
    img = img.convert("RGB")

    # Resize to model's input shape
    img = img.resize(input_shape)

    # Convert to numpy array
    img = np.array(img)

    # Normalize pixel values
    img = img / 255.0  

    # Ensure correct shape (1, height, width, 3)
    img = img.reshape(1, input_shape[0], input_shape[1], 3)
    
    return img

@login_required
def predict(request):
    if request.method == "POST":
        try:
            model_key = request.POST.get("model_key")
            imagefile = request.FILES.get("imagefile")

            if not model_key or not imagefile:
                return render(request, "prediction.html", {"error": "Model key and image file are required."})

            img_array = preprocess_image(imagefile, model_key)
            model = models[model_key]["model"]
            class_labels = models[model_key]["class_labels"]

            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100
            # prediction_result = f"{confidence:.2f}% Confidence: {class_labels[class_index]}"
            prediction_result = f"{class_labels[class_index]}"

            prediction = History.objects.create(
                user=request.user,
                model_key=model_key,
                result=prediction_result,
                image=imagefile,
            )
            return render(request, "prediction.html", {
                "prediction": prediction_result,
                "image_url": prediction.image.url,
                "user": request.user,
                "model_key": model_key
            })

        except Exception as e:
            return render(request, "prediction.html", {"error": str(e)})

    return render(request, "prediction.html")