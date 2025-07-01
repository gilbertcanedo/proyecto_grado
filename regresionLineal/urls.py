from django.urls import path
from . import views

urlpatterns =  [

    
    path('prediccionPG/', views.prediccion_pg, name='prediccionPG'),
    path('prediccionPFC/', views.prediccion_pfc, name='prediccionPFC'),
    path('prediccionCP/', views.prediccion_cp, name='prediccionCP'),
    
]                         
