# Generated by Django 5.2.1 on 2025-06-08 19:08

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('transfusiones', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='transfusion',
            name='cama',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='combsdirecto',
            name='nombre',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='combsindirecto',
            name='nombre',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='comb_directo',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, to='transfusiones.combsdirecto'),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='comb_indirecto',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, to='transfusiones.combsindirecto'),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='material1',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='material2',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='material3',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='material4',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='transfusion',
            name='material5',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
