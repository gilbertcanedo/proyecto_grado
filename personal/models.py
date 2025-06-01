from django.db import models

# Create your models here.
class Responsable(models.Model):
    nombre = models.CharField(max_length=100)
    apellido_paterno = models.CharField(max_length=100)
    apellido_materno = models.CharField(max_length=100)
    grado_academico = models.CharField(max_length=100)

class Medico(models.Model):
    nombre = models.CharField(max_length=100)
    apellido_paterno = models.CharField(max_length=100)
    apellido_materno = models.CharField(max_length=100)
    grado_academico = models.CharField(max_length=100)
