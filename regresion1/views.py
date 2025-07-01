import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- Cargar Modelo y Datos -------------------

PAQUETE = joblib.load('modelo/modelo_regresion_multipla_completo2.pkl')
MODELO = PAQUETE["modelo"]
FECHAS_HIST = pd.to_datetime(PAQUETE["fechas_hist"])
TRANSF_HIST = np.array(PAQUETE["transfusiones_hist"])
FECHAS_FUTURAS = pd.to_datetime(PAQUETE["fechas_futuras"])
COLUMNAS = PAQUETE["columnas"]

MESES_A_PREDECIR = 48

# ------------------- Generar Variables Futuras -------------------

df_futuro = pd.DataFrame({
    'Tendencia': np.arange(len(FECHAS_HIST) + 1, len(FECHAS_HIST) + MESES_A_PREDECIR + 1),
    'Mes': FECHAS_FUTURAS.month.astype(str)
})
X_futuro = pd.get_dummies(df_futuro, drop_first=True)

# Alineamos columnas
for col in COLUMNAS:
    if col not in X_futuro.columns:
        X_futuro[col] = 0
X_futuro = X_futuro[COLUMNAS]

# Predicciones
PREDICCIONES = MODELO.predict(X_futuro)

# ------------------- Métricas adicionales -------------------

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mean_absolute_scaled_error(y_true, y_pred):
    n = len(y_true)
    d = np.abs(np.diff(y_true)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

# ------------------- Vista de Predicción -------------------

def prediccion_transfusiones1(request):
    resultado = None
    error = None
    mae = rmse = r2 = None
    mape = smape = mase = None
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
                resultado = int(np.round(PREDICCIONES[idx]))

                # Entrenamiento interno histórico
                df_hist = pd.DataFrame({
                    'Tendencia': np.arange(1, len(FECHAS_HIST) + 1),
                    'Mes': FECHAS_HIST.month.astype(str)
                })
                X_hist = pd.get_dummies(df_hist, drop_first=True)

                for col in COLUMNAS:
                    if col not in X_hist.columns:
                        X_hist[col] = 0
                X_hist = X_hist[COLUMNAS]

                pred_train = MODELO.predict(X_hist)

                mae = round(mean_absolute_error(TRANSF_HIST, pred_train), 2)
                rmse = round(np.sqrt(mean_squared_error(TRANSF_HIST, pred_train)), 2)
                r2 = round(r2_score(TRANSF_HIST, pred_train), 2)
                mape = round(mean_absolute_percentage_error(TRANSF_HIST, pred_train), 2)
                smape = round(symmetric_mean_absolute_percentage_error(TRANSF_HIST, pred_train), 2)
                mase = round(mean_absolute_scaled_error(TRANSF_HIST, pred_train), 2)

                datos_historicos = [
                    {"fecha": f.strftime("%Y-%m"), "valor": int(v)}
                    for f, v in zip(FECHAS_HIST, TRANSF_HIST)
                ]

                datos_predicciones = [
                    {"fecha": f.strftime("%Y-%m"), "valor": float(p)}
                    for f, p in zip(FECHAS_FUTURAS, PREDICCIONES)
                ]
            else:
                error = "La fecha ingresada está fuera del rango permitido."

        except Exception as e:
            error = f"Error: {e}"

    return render(request, 'regresion1/regresion1.html', {
        "resultado": resultado,
        "error": error,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "mase": mase,
        "datos_historicos": datos_historicos,
        "datos_predicciones": datos_predicciones,
        "mes_texto": mes_texto,
        "anio_seleccionado": anio_seleccionado
    })