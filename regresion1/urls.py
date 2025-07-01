from django.urls import path
from . import views

urlpatterns =  [

    
    path('regresion2/', views.prediccion_transfusiones1, name='regresion2'),
    
]                         
