from django.db import models

# Create your models here.
class TipoSangre(models.Model):
    nombre = models.CharField(max_length=50)
    sigla = models.CharField(max_length=5)
    prueba1 = models.CharField(max_length=5)
    prueba2 = models.CharField(max_length=5)
    prueba3 = models.CharField(max_length=5)
    prueba4 = models.CharField(max_length=5)
    prueba5 = models.CharField(max_length=5)
    prueba6 = models.CharField(max_length=5)
    prueba7 = models.CharField(max_length=5)
    prueba8 = models.CharField(max_length=5)

    def __str__(self):
        return f"{self.sigla} ({self.nombre})"

class TipoHemocomponente(models.Model):
    nombre = models.CharField(max_length=50)
    sigla = models.CharField(max_length=10)
    cantidad = models.PositiveIntegerField()
    descripcion = models.TextField()

class Hemocomponente(models.Model):
    tipo_hemo = models.ForeignKey(TipoHemocomponente, on_delete=models.PROTECT)
    numero_bolsa = models.CharField(max_length=50)
    numero_tubuladura = models.CharField(max_length=50)
    numero_barra = models.CharField(max_length=50)
    tipo_sangre = models.ForeignKey(TipoSangre, on_delete=models.PROTECT)
    solfis = models.CharField(max_length=5)
    liss = models.CharField(max_length=5)
    albumina = models.CharField(max_length=5)
    mayor = models.CharField(max_length=5)
    menor = models.CharField(max_length=5)
    compatibilidad = models.CharField(max_length=5)

    def __str__(self):
        return f"{self.tipo_hemo.sigla} - {self.numero_bolsa}"