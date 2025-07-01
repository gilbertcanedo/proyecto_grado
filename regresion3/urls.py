from django.urls import path
from . import views

urlpatterns =  [

    
    path('regresion4/', views.prediccion_transfusiones3, name='regresion4'),
    
]                         
