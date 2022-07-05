from django.urls import path
from . import views

app_name = 'ocr'

urlpatterns = [
    path('', views.home, name='home'),
    path('scan/', views.scan, name='scan'),
    path('scan/cam/', views.cam, name='cam'),
    path('scan/get_cam', views.get_cam, name='get_cam'),
]
