from django.urls import path, include
from . import views

app_name = 'car'

urlpatterns = [
    path('enter/', views.enter, name='enter'),
    path('enter_call/', views.enter_call, name='enter_call'),
]
