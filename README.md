# Predicción inmobiliaria — Prefectura de Tokio

**Proyecto Integrador — Inteligencia Artificial y Aprendizaje Automático I (2026)**
Licenciatura en Ciencia de Datos · Pontificia Universidad Católica Argentina, Rosario

Aplicación web que despliega dos modelos de machine learning entrenados sobre el registro
público de transacciones inmobiliarias del **MLIT** (Ministerio de Tierra, Infraestructura,
Transporte y Turismo de Japón) para la Prefectura de Tokio, período 2005–2025 (343.995 operaciones).

## Estructura del repositorio

```
.
├── app.py                       # Aplicación Streamlit
├── preparar_despliegue.py       # Copia los modelos y exporta metadatos
├── requirements.txt
├── README.md
├── models/                      # Artefactos serializados (generados, no versionados)
│   ├── tp2_mejor_regresor.joblib
│   ├── tp3_mejor_clasificador.joblib
│   └── metadatos.json
└── notebooks/
    ├── TP1_EDA_Tokyo.ipynb            # Análisis exploratorio y preparación de datos
    ├── TP2_Regresion_Tokyo.ipynb      # Regresión sobre el precio de transacción
    └── TP3_Clasificacion_Tokyo.ipynb  # Clasificación del sistema constructivo
```

## Como ejecutarlo

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone <url-del-repo>
cd <carpeta-del-repo>
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generar los modelos

Los artefactos `.joblib` no se versionan en el repositorio por su tamaño. Es necesario ejecutar
los notebooks en orden, con el CSV original del MLIT disponible localmente:

```
TP1 → outputs/processed/tokyo_tp1_procesado.csv
TP2 → outputs/models/tp2_mejor_regresor.joblib
TP3 → outputs/models/tp3_mejor_clasificador.joblib
```

TP2 y TP3 son independientes entre sí; ambos consumen únicamente la salida del TP1.

### 3. Preparar los artefactos de despliegue

```bash
python preparar_despliegue.py
```

Copia los modelos a `models/` y extrae del dataset las categorías reales observadas en el
entrenamiento (wards, zonificación, formas de lote, etc.) hacia `models/metadatos.json`, de modo
que los desplegables del formulario ofrezcan exclusivamente valores dentro del dominio del modelo.

### 4. Levantar la aplicación

```bash
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), conectar la cuenta y seleccionar el repositorio.
3. Indicar `app.py` como archivo principal de la aplicación.

**Consideración sobre el tamaño de los modelos.** Los artefactos `.joblib` pueden superar el
límite práctico de un repositorio Git estándar, en particular el correspondiente a Random Forest.
Alternativas recomendadas:

- Versionar los artefactos con [Git LFS](https://git-lfs.github.com/).
- Publicarlos como *release asset* de GitHub y descargarlos al iniciar la aplicación.
- Reducir `n_estimators` en el reentrenamiento para disminuir el tamaño del artefacto serializado.

En este proyecto, el modelo elegido para ambas tareas es XGBoost, cuyo artefacto serializado es
considerablemente más liviano que el de un Random Forest equivalente y en general no presenta
inconvenientes de tamaño en el despliegue.

## Resultados de los modelos

### Regresión — precio de transacción (TP2)

| Modelo | R² (log) test | R² (¥) test | RMSE | MAE | MAPE |
|---|---|---|---|---|---|
| Regresión Lineal | 0,724 | 0,100 | ¥690.425.179 | ¥47.182.148 | 164,3% |
| Random Forest | 0,705 | 0,077 | ¥699.123.745 | ¥43.225.680 | 195,9% |
| **XGBoost** | **0,848** | **0,267** | **¥622.950.151** | **¥29.490.866** | **140,1%** |

La brecha entre el R² en escala logarítmica y el R² en yenes se explica porque la transformación
inversa (`expm1`) amplifica exponencialmente el error en la cola de precios altos. El modelo es
adecuado para estimar el orden de magnitud de una operación típica, no para tasar con precisión:
un MAPE del 140% implica que la estimación puede desviarse en más del doble del valor real.

### Clasificación — estructura edilicia (TP3)

El desbalance de clases es severo: `W` (madera) concentra el 80,8% de los 224.622 edificios,
seguida por `S` (7,5%), `RC` (7,45%), `Otros` (3,0%) y `SRC` (1,2%) — una relación de 65,6:1 entre
la clase mayoritaria y la minoritaria. Por eso la métrica de referencia es el **F1 macro** y no la
accuracy: un clasificador trivial que prediga siempre `W` alcanzaría ~81% de accuracy sin detectar
ninguna otra estructura.

El proyecto compara cinco modelos en dos bloques: tres clasificadores individuales (Sección B) y
dos ensambles (Sección D.2), incorporados en una etapa posterior del pipeline experimental.

| Modelo | Bloque | F1 macro | F1 ponderado | Accuracy | Kappa | ROC-AUC (ovr) |
|---|---|---|---|---|---|---|
| Regresión Logística | Individual | 0,492 | 0,778 | 0,725 | 0,408 | 0,903 |
| Árbol de Decisión | Individual | 0,479 | 0,762 | 0,703 | 0,395 | 0,891 |
| Naive Bayes | Individual | 0,121 | 0,258 | 0,175 | 0,038 | 0,579 |
| Random Forest | Ensamble | 0,554 | 0,812 | 0,772 | 0,485 | N/D* |
| **XGBoost** | Ensamble | **0,568** | **0,876** | **0,892** | **0,636** | N/D* |

\* El ROC-AUC one-vs-rest se calculó sobre los tres clasificadores individuales de la Sección B;
los ensambles se incorporaron al registro de resultados en una etapa posterior y no cuentan con
este valor recalculado.

Comparando el Árbol de Decisión con su versión ensamblada (Random Forest), el aporte aislado del
*bagging* es de +0,075 en F1 macro. Comparando el mejor modelo individual (Regresión Logística)
con el mejor ensamble (XGBoost), la ganancia total es de +0,076.

## Limitaciones

- Datos **autoreportados** por las partes de la operación: posible subreporte y redondeo.
- Cobertura exclusiva de la Prefectura de Tokio; **no generaliza** a otras prefecturas.
- El registro no incluye calidad constructiva, estado de conservación ni condiciones
  particulares de cada negociación.
- Superficies mayores a 2.000 m² están **censuradas** en la fuente original.
- `Building : Structure` es multi-etiqueta en el origen (por ejemplo, `"RC, W"`); se simplificó
  al sistema constructivo principal, agrupando las clases marginales (`LS`, `B`) en `Otros`.
- El clasificador exhibe bajo recall en las clases minoritarias (`Otros`: 0,048; `SRC`: 0,454),
  compensado parcialmente por los otros modelos, que las detectan mejor a costa de una accuracy
  global menor.

Este proyecto tiene fines **académicos y demostrativos**. No constituye una tasación profesional
ni debe utilizarse como base para decisiones de inversión.

## Fuente de datos

MLIT — *Real Estate Transaction Price Information*
<https://www.land.mlit.go.jp/webland_english/servlet/MainServlet>
