from django.urls import path
from . import views

app_name = 'ocr'

urlpatterns = [
    path('scan/', views.scan, name='scan'),
    path('scan/mycam', views.mycam, name='mycam'),
]
