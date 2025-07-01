import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- Cargar Modelo y Datos -------------------

PAQUETE = joblib.load('modelo/modelo_regresion_lineal_pg_completo.pkl')
MODELO = PAQUETE["modelo"]
FECHAS_HIST = pd.to_datetime(PAQUETE["fechas_hist"])
PG_HIST = np.array(PAQUETE["pg_hist"])
FECHAS_FUTURAS = pd.to_datetime(PAQUETE["fechas_futuras"])
PREDICCIONES_FUTURAS = np.array(PAQUETE["predicciones_futuras"])

# ------------------- Vista para PG -------------------

def prediccion_pg(request):
    resultado = None
    error = None
    mae = rmse = r2 = None
    datos_historicos = []
    datos_predicciones = []
    mes_texto = None
    anio_seleccionado = None

    if request.method == "POST":
        try:
            mes = int(request.POST.get("mes"))
            anio = int(request.POST.get("anio"))
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes:02d}-01")

            nombre_meses = [
                'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
            ]

            mes_texto = nombre_meses[mes - 1]
            anio_seleccionado = anio

            if fecha_objetivo in FECHAS_FUTURAS:
                idx = FECHAS_FUTURAS.get_loc(fecha_objetivo)
                resultado = int(np.round(PREDICCIONES_FUTURAS[idx]))

                # Generar Tendencia histórica
                tendencia_hist = np.arange(1, len(FECHAS_HIST) + 1).reshape(-1, 1)
                pred_train = MODELO.predict(tendencia_hist)

                # Métricas sobre histórico
                mae = round(mean_absolute_error(PG_HIST, pred_train), 2)
                rmse = round(np.sqrt(mean_squared_error(PG_HIST, pred_train)), 2)
                r2 = round(r2_score(PG_HIST, pred_train), 2)

                datos_historicos = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(v)}
                    for f, v in zip(FECHAS_HIST, PG_HIST)
                ]

                datos_predicciones = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(p)}
                    for f, p in zip(FECHAS_FUTURAS, PREDICCIONES_FUTURAS)
                ]
            else:
                error = "La fecha ingresada está fuera del rango permitido."

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'regresionLineal/hemocomponentePG.html', {
        "resultado": resultado,
        "error": error,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "datos_historicos": datos_historicos,
        "datos_predicciones": datos_predicciones,
        "mes_texto": mes_texto,
        "anio_seleccionado": anio_seleccionado
    })

# ---------------- Cargar Modelo PFC----------------

PAQUETE_PFC = joblib.load('modelo/modelo_regresion_lineal_pfc_completo.pkl')
MODELO_PFC = PAQUETE_PFC["modelo"]
FECHAS_HIST_PFC = pd.to_datetime(PAQUETE_PFC["fechas_hist"])
PFC_HIST = np.array(PAQUETE_PFC["pfc_hist"])
FECHAS_FUTURAS_PFC = pd.to_datetime(PAQUETE_PFC["fechas_futuras"])
PREDICCIONES_FUTURAS_PFC = np.array(PAQUETE_PFC["predicciones_futuras"])

# ---------------- Vista PFC----------------

def prediccion_pfc(request):
    resultado = None
    error = None
    datos_historicos = []
    datos_predicciones = []
    mes_texto = None
    anio_seleccionado = None

    mae_train = rmse_train = r2_train = None
    mae_test = rmse_test = r2_test = None

    if request.method == "POST":
        try:
            mes = int(request.POST.get("mes"))
            anio = int(request.POST.get("anio"))
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes:02d}-01")

            nombre_meses = [
                'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
            ]

            mes_texto = nombre_meses[mes - 1]
            anio_seleccionado = anio

            if fecha_objetivo in FECHAS_FUTURAS_PFC:
                idx = FECHAS_FUTURAS_PFC.get_loc(fecha_objetivo)
                resultado = int(np.round(PREDICCIONES_FUTURAS_PFC[idx]))

                # Tendencias
                tendencia_total = np.arange(1, len(FECHAS_HIST_PFC) + 1).reshape(-1, 1)
                tendencia_test = tendencia_total[int(len(tendencia_total)*0.8):]

                y_pred_total = MODELO_PFC.predict(tendencia_total)
                y_pred_test = MODELO_PFC.predict(tendencia_test)

                # Métricas
                mae_train = round(mean_absolute_error(PFC_HIST[:int(len(PFC_HIST)*0.8)], y_pred_total[:int(len(PFC_HIST)*0.8)]), 2)
                rmse_train = round(np.sqrt(mean_squared_error(PFC_HIST[:int(len(PFC_HIST)*0.8)], y_pred_total[:int(len(PFC_HIST)*0.8)])), 2)
                r2_train = round(r2_score(PFC_HIST[:int(len(PFC_HIST)*0.8)], y_pred_total[:int(len(PFC_HIST)*0.8)]), 2)

                mae_test = round(mean_absolute_error(PFC_HIST[int(len(PFC_HIST)*0.8):], y_pred_test), 2)
                rmse_test = round(np.sqrt(mean_squared_error(PFC_HIST[int(len(PFC_HIST)*0.8):], y_pred_test)), 2)
                r2_test = round(r2_score(PFC_HIST[int(len(PFC_HIST)*0.8):], y_pred_test), 2)

                # Datos para gráfica
                datos_historicos = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(v)}
                    for f, v in zip(FECHAS_HIST_PFC, PFC_HIST)
                ]

                datos_predicciones = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(p)}
                    for f, p in zip(FECHAS_FUTURAS_PFC, PREDICCIONES_FUTURAS_PFC)
                ]
            else:
                error = "La fecha está fuera del rango de predicción."

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'regresionLineal/hemocomponentePFC.html', {
        "resultado": resultado,
        "error": error,
        "datos_historicos": datos_historicos,
        "datos_predicciones": datos_predicciones,
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "mes_texto": mes_texto,
        "anio_seleccionado": anio_seleccionado
    })


# ---------------- Cargar Modelo CP ----------------

PAQUETE_CP = joblib.load('modelo/modelo_regresion_lineal_cp_completo.pkl')
MODELO_CP = PAQUETE_CP["modelo"]
FECHAS_HIST_CP = pd.to_datetime(PAQUETE_CP["fechas_hist"])
CP_HIST = np.array(PAQUETE_CP["cp_hist"])
FECHAS_FUTURAS_CP = pd.to_datetime(PAQUETE_CP["fechas_futuras"])
PREDICCIONES_FUTURAS_CP = np.array(PAQUETE_CP["predicciones_futuras"])

# ---------------- Vista CP----------------

def prediccion_cp(request):
    resultado = None
    error = None
    datos_historicos = []
    datos_predicciones = []
    mes_texto = None
    anio_seleccionado = None

    mae_train = rmse_train = r2_train = None
    mae_test = rmse_test = r2_test = None

    if request.method == "POST":
        try:
            mes = int(request.POST.get("mes"))
            anio = int(request.POST.get("anio"))
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes:02d}-01")

            nombre_meses = [
                'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
            ]

            mes_texto = nombre_meses[mes - 1]
            anio_seleccionado = anio

            if fecha_objetivo in FECHAS_FUTURAS_CP:
                idx = FECHAS_FUTURAS_CP.get_loc(fecha_objetivo)
                resultado = int(np.round(PREDICCIONES_FUTURAS_CP[idx]))

                tendencia_total = np.arange(1, len(FECHAS_HIST_CP) + 1).reshape(-1, 1)
                tendencia_test = tendencia_total[int(len(tendencia_total)*0.8):]

                y_pred_total = MODELO_CP.predict(tendencia_total)
                y_pred_test = MODELO_CP.predict(tendencia_test)

                mae_train = round(mean_absolute_error(CP_HIST[:int(len(CP_HIST)*0.8)], y_pred_total[:int(len(CP_HIST)*0.8)]), 2)
                rmse_train = round(np.sqrt(mean_squared_error(CP_HIST[:int(len(CP_HIST)*0.8)], y_pred_total[:int(len(CP_HIST)*0.8)])), 2)
                r2_train = round(r2_score(CP_HIST[:int(len(CP_HIST)*0.8)], y_pred_total[:int(len(CP_HIST)*0.8)]), 2)

                mae_test = round(mean_absolute_error(CP_HIST[int(len(CP_HIST)*0.8):], y_pred_test), 2)
                rmse_test = round(np.sqrt(mean_squared_error(CP_HIST[int(len(CP_HIST)*0.8):], y_pred_test)), 2)
                r2_test = round(r2_score(CP_HIST[int(len(CP_HIST)*0.8):], y_pred_test), 2)

                datos_historicos = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(v)}
                    for f, v in zip(FECHAS_HIST_CP, CP_HIST)
                ]

                datos_predicciones = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(p)}
                    for f, p in zip(FECHAS_FUTURAS_CP, PREDICCIONES_FUTURAS_CP)
                ]
            else:
                error = "La fecha está fuera del rango de predicción."

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'regresionLineal/hemocomponenteCP.html', {
        "resultado": resultado,
        "error": error,
        "datos_historicos": datos_historicos,
        "datos_predicciones": datos_predicciones,
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "mes_texto": mes_texto,
        "anio_seleccionado": anio_seleccionado
    })