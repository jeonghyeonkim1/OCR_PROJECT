from django.urls import path, include
from torch import view_as_complex
from . import views

app_name = 'ocr'

urlpatterns = [
    path('scan/', views.scan, name='scan'),
    path('scan/mycam', views.mycam, name='mycam'),
]
