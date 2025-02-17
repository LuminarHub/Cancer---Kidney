from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm

class  UserForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username','email','password1','password2']


class LogForm(forms.Form):
    username=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.75rem; "}))
    password=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.75rem; "}))
