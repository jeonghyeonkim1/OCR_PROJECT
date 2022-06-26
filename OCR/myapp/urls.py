
from django.urls import path, include
from . import views

urlpatterns = [
    path('ocr/', views.ocr),
    path('ocr/video', views.video),
]
