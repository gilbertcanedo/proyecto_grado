from django.db import models
from sangre.models import TipoSangre
# Create your models here.
class Genero(models.Model):
    nombre = models.CharField(max_length=50)
    sigla = models.CharField(max_length=10)

class Paciente(models.Model):
    nombre = models.CharField(max_length=100)
    apellido_paterno = models.CharField(max_length=100)
    apellido_materno = models.CharField(max_length=100)
    ci = models.CharField(max_length=20, unique=True)
    edad = models.PositiveIntegerField()
    numero_factura = models.CharField(max_length=50)
    tipo_sangre = models.ForeignKey(TipoSangre, on_delete=models.PROTECT)
    genero = models.ForeignKey(Genero, on_delete=models.PROTECT)
