from django.contrib import admin
from .models import GradoEmergencia,Unidad,CombsDirecto,CombsIndirecto,Diagnostico,Observacion,Transfusion
# Register your models here.
admin.site.register(GradoEmergencia)
admin.site.register(Unidad)
admin.site.register(CombsDirecto)
admin.site.register(CombsIndirecto)
admin.site.register(Diagnostico)
admin.site.register(Observacion)
admin.site.register(Transfusion)