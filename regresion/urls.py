from django.urls import path
from . import views

urlpatterns =  [

    path('regresion1/', views.prediccion_transfusiones, name='regresion1'),
    path('prediccion/', views.prediccion, name='prediccion'),
    
]                         
