from django.contrib import admin
from .models import TipoSangre,TipoHemocomponente, Hemocomponente
# Register your models here.
admin.site.register(TipoSangre)
admin.site.register(TipoHemocomponente)
admin.site.register(Hemocomponente)
