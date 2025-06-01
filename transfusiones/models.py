from django.db import models
from pacientes.models import Paciente
from sangre.models import Hemocomponente
#from .models import CombinacionDirecta, CombinacionIndirecta, Unidad, Diagnostico, Observacion
from personal.models import Responsable, Medico

# Create your models here.
class GradoEmergencia(models.Model):
    numero_emergencia = models.PositiveIntegerField()
    nombre = models.CharField(max_length=100)

class Unidad(models.Model):
    nombre = models.CharField(max_length=100)
    sigla = models.CharField(max_length=10)
    grado_emergencia = models.ForeignKey(GradoEmergencia, on_delete=models.PROTECT)

class CombsDirecto(models.Model):
    nombre = models.CharField(max_length=100)
    sigla = models.CharField(max_length=10)

class CombsIndirecto(models.Model):
    nombre = models.CharField(max_length=100)
    sigla = models.CharField(max_length=10)

class Diagnostico(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField()

class Observacion(models.Model):
    sigla = models.CharField(max_length=10)
    descripcion = models.TextField()

class Transfusion(models.Model):
    fecha = models.DateField()
    hora = models.TimeField()
    paciente = models.ForeignKey(Paciente, on_delete=models.PROTECT)
    hemocomponente = models.ForeignKey(Hemocomponente, on_delete=models.PROTECT)
    unidad = models.ForeignKey(Unidad, on_delete=models.PROTECT)
    comb_directo = models.ForeignKey(CombsDirecto, on_delete=models.PROTECT)
    comb_indirecto = models.ForeignKey(CombsIndirecto, on_delete=models.PROTECT)
    diagnostico = models.ForeignKey(Diagnostico, on_delete=models.PROTECT)
    material1 = models.CharField(max_length=5)
    material2 = models.CharField(max_length=5)
    material3 = models.CharField(max_length=5)
    material4 = models.CharField(max_length=5)
    material5 = models.CharField(max_length=5)
    responsable = models.ForeignKey(Responsable, on_delete=models.PROTECT)
    medico = models.ForeignKey(Medico, on_delete=models.PROTECT)
    observacion = models.ForeignKey(Observacion, on_delete=models.PROTECT)
    numero_tratamiento1 = models.CharField(max_length=50)
    numero_tratamiento2 = models.CharField(max_length=50)

    def __str__(self):
        return f"Transfusión {self.id} - {self.paciente}"
