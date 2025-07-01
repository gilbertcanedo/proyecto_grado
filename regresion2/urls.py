from django.urls import path
from . import views

urlpatterns =  [

    
    path('regresion3/', views.prediccion_transfusiones2, name='regresion3'),
    
]                         
