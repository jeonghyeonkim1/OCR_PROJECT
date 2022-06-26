
from django.urls import path, include
from torch import view_as_complex
from . import views

app_name = 'ocr'

urlpatterns = [
    path('enter/', views.enter, name='enter'),
    path('enter_call/', views.enter_call, name='enter_call'),
    path('scan/', views.scan, name='scan'),
    path('scan/mycam', views.mycam, name='mycam'),
]
