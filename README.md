# Predicción inmobiliaria — Prefectura de Tokio

Proyecto Integrador de **Inteligencia Artificial y Aprendizaje Automático I** (2026)
Licenciatura en Ciencia de Datos · UCA Rosario

Aplicación web que despliega dos modelos de machine learning entrenados sobre el registro
público de transacciones inmobiliarias del **MLIT** (Ministerio de Tierra, Infraestructura,
Transporte y Turismo de Japón), para la Prefectura de Tokio en el período 2005–2025
(~344.000 operaciones).

## Qué hace

| Pestaña | Tarea | Modelo | Métrica principal |
|---|---|---|---|
| 💴 Estimar precio | Regresión | XGBoost sobre `log1p(precio)` | R² (log) = 0,848 |
| 🏗️ Clasificar estructura | Clasificación multiclase | XGBoost | F1 macro = 0,566 |

Ambas pestañas comunican explícitamente la **incertidumbre** de la predicción: la de precio
muestra un intervalo y advierte cuando el valor cae en la cola alta de la distribución; la de
estructura muestra las probabilidades por clase y señala cuándo la predicción corresponde a una
clase minoritaria.

## Estructura del proyecto

```
.
├── app.py                       # Aplicación Streamlit
├── preparar_despliegue.py       # Copia modelos y exporta metadatos
├── requirements.txt
├── README.md
├── models/                      # Artefactos (generados, no versionados)
│   ├── tp2_mejor_regresor.joblib
│   ├── tp3_mejor_clasificador.joblib
│   └── metadatos.json
└── notebooks/
    ├── TP1_EDA_Tokyo.ipynb            # EDA y preparación de datos
    ├── TP2_Regresion_Tokyo.ipynb      # Regresión sobre el precio
    └── TP3_Clasificacion_Tokyo.ipynb  # Clasificación de la estructura
```

## Cómo ejecutarlo localmente

**1. Clonar e instalar dependencias**

```bash
git clone <url-del-repo>
cd <carpeta-del-repo>
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generar los modelos**

Los `.joblib` no se versionan (pesan demasiado para Git). Hay que ejecutar los notebooks en
orden, con el CSV original del MLIT disponible:

```
TP1 → outputs/processed/tokyo_tp1_procesado.csv
TP2 → outputs/models/tp2_mejor_regresor.joblib
TP3 → outputs/models/tp3_mejor_clasificador.joblib
```

El TP2 y el TP3 son independientes entre sí: ambos consumen la salida del TP1.

**3. Preparar los artefactos**

```bash
python preparar_despliegue.py
```

Copia los modelos a `models/` y extrae del dataset las categorías reales de cada variable
(wards, zonificación, formas de lote, etc.) hacia `models/metadatos.json`, de modo que los
desplegables del formulario ofrezcan exactamente las opciones que el modelo vio al entrenar.

**4. Levantar la app**

```bash
streamlit run app.py
```

Queda disponible en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), conectar la cuenta y elegir el repo.
3. Indicar `app.py` como archivo principal.

**Sobre los modelos en el deploy.** Los `.joblib` pueden superar el límite práctico de Git
(especialmente el Random Forest). Opciones:

- Usar [Git LFS](https://git-lfs.github.com/) para versionarlos.
- Publicarlos como *release asset* en GitHub y descargarlos al iniciar la app.
- Reentrenar con `n_estimators` más bajo para reducir el tamaño del artefacto.

Si el modelo elegido fue XGBoost (el caso de este proyecto), el archivo es considerablemente
más chico que un Random Forest equivalente y suele entrar sin problema.

## Resultados de los modelos

### Regresión — precio de transacción (TP2)

| Modelo | R² (log) test | R² (¥) test | RMSE | MAE | MAPE |
|---|---|---|---|---|---|
| Regresión Lineal | 0,724 | 0,100 | ¥690.425.179 | ¥47.182.148 | 164,3% |
| Random Forest | 0,705 | 0,077 | ¥699.123.745 | ¥43.225.680 | 195,9% |
| **XGBoost** | **0,848** | **0,267** | **¥622.950.151** | **¥29.490.866** | **140,1%** |

La brecha entre el R² logarítmico y el R² en yenes se explica porque la transformación inversa
(`expm1`) amplifica exponencialmente el error en la cola de precios altos. El modelo es útil
para estimar el orden de magnitud de una operación típica, no para tasar con precisión.

### Clasificación — estructura edilicia (TP3)

| Modelo | F1 macro | F1 ponderado | Accuracy | Kappa | ROC-AUC |
|---|---|---|---|---|---|
| Regresión Logística | 0,493 | 0,778 | 0,725 | 0,409 | 0,903 |
| Random Forest | 0,554 | 0,812 | 0,772 | 0,485 | 0,932 |
| **XGBoost** | **0,566** | **0,876** | **0,892** | **0,635** | **0,948** |

Distribución de clases: `W` 80,8% · `S` 7,5% · `RC` 7,45% · `Otros` 3,0% · `SRC` 1,2%
(relación 65,6:1 entre la mayoritaria y la minoritaria). Por eso la métrica de referencia es el
**F1 macro** y no la accuracy: un clasificador trivial que prediga siempre `W` alcanzaría ~81%
de accuracy sin detectar ninguna otra estructura.

## Limitaciones

- Datos **autoreportados** por las partes de la operación: posible subreporte y redondeo.
- Cobertura exclusiva de la Prefectura de Tokio; **no generaliza** a otras prefecturas.
- El registro no incluye calidad constructiva, estado de conservación ni condiciones
  particulares de cada negociación.
- Superficies mayores a 2.000 m² están **censuradas** en la fuente original.
- `Building : Structure` es multi-etiqueta en el origen (`"RC, W"`); se simplificó al sistema
  constructivo principal, agrupando las clases raras (`LS`, `B`) en `Otros`.

Uso **académico y demostrativo**. No constituye una tasación profesional ni debe usarse como
base para decisiones de inversión.

## Fuente de datos

MLIT — *Real Estate Transaction Price Information*
<https://www.land.mlit.go.jp/webland_english/servlet/MainServlet>
