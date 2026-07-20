"""
TP4 — Despliegue en producción
Proyecto Integrador · Inteligencia Artificial y Aprendizaje Automático I · 2026
UCA · Licenciatura en Ciencia de Datos

App Streamlit que despliega los dos modelos del proyecto:
  - Pestaña 1: estimación del precio de transacción (regresor del TP2).
  - Pestaña 2: clasificación de la estructura edilicia (clasificador del TP3).

Ambos modelos se cargan como Pipeline serializado (preprocesamiento + modelo) desde .joblib,
de modo que la app no replica ninguna lógica de transformación: recibe los datos crudos del
formulario y el pipeline se encarga del resto.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tokio · Predicción inmobiliaria",
    page_icon="🏙️",
    layout="wide",
)

DIR_BASE = Path(__file__).parent
RUTA_REGRESOR = DIR_BASE / "models" / "tp2_mejor_regresor.joblib"
RUTA_CLASIFICADOR = DIR_BASE / "models" / "tp3_mejor_clasificador.joblib"
RUTA_METADATOS = DIR_BASE / "models" / "metadatos.json"

# Error típico del regresor en escala log (desvío de residuos medido en el TP2).
# Se usa para construir el intervalo de confianza aproximado.
SIGMA_RESIDUOS_LOG = 0.3693

# Umbral por encima del cual la estimación se considera poco confiable: el TP2 mostró que el
# error crece fuertemente en la cola de precios altos (heterocedasticidad remanente).
UMBRAL_ALERTA_YEN = 500_000_000


# --------------------------------------------------------------------------------------------
# Carga de artefactos (cacheada: se ejecuta una sola vez por sesión)
# --------------------------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    """Carga los dos pipelines serializados y los metadatos del dataset."""
    faltantes = [str(r.name) for r in (RUTA_REGRESOR, RUTA_CLASIFICADOR) if not r.exists()]
    if faltantes:
        return None, None, None, faltantes

    regresor = joblib.load(RUTA_REGRESOR)
    clasificador = joblib.load(RUTA_CLASIFICADOR)

    # Los modelos se entrenaron en GPU (device="cuda"), pero la app predice de a una fila
    # sobre DataFrames en CPU. Ese desajuste puede hacer caer el proceso en Windows.
    # Forzamos CPU: además es lo que corresponde para desplegar en Streamlit Cloud (sin GPU).
    for artefacto in (regresor, clasificador):
        modelo = artefacto["pipeline"].named_steps.get("model")
        if modelo is not None and type(modelo).__name__.startswith("XGB"):
            try:
                modelo.set_params(device="cpu")
                if hasattr(modelo, "get_booster"):
                    modelo.get_booster().set_param({"device": "cpu"})
            except Exception:
                pass

    metadatos = {}
    if RUTA_METADATOS.exists():
        metadatos = json.loads(RUTA_METADATOS.read_text(encoding="utf-8"))

    return regresor, clasificador, metadatos, []


regresor, clasificador, metadatos, faltantes = cargar_modelos()

if faltantes:
    st.error(
        "No se encontraron los modelos entrenados: "
        + ", ".join(faltantes)
        + ".\n\nEjecutá los notebooks del TP2 y TP3 y copiá los archivos `.joblib` "
        "generados en `outputs/models/` a la carpeta `models/` de esta app."
    )
    st.stop()


# --------------------------------------------------------------------------------------------
# Valores por defecto de los formularios
# --------------------------------------------------------------------------------------------
# Se leen de metadatos.json si existe (generado junto con los modelos); si no, se usan
# valores de respaldo tomados del EDA del TP1.
WARDS = metadatos.get("wards", [
    "Chiyoda", "Chuo", "Minato", "Shinjuku", "Bunkyo", "Taito", "Sumida", "Koto",
    "Shinagawa", "Meguro", "Ota", "Setagaya", "Shibuya", "Nakano", "Suginami",
    "Toshima", "Kita", "Arakawa", "Itabashi", "Nerima", "Adachi", "Katsushika",
    "Edogawa", "Hachioji", "Machida", "Fuchu", "Kodaira",
])
ZONAS = metadatos.get("zonas", [
    "Residential Area", "Commercial Area", "Industrial Area", "Potential Residential Area",
])
CITY_PLANNING = metadatos.get("city_planning", [
    "1 Exc Low", "1 Exc Med", "1 Res", "2 Exc Low", "2 Exc Med", "2 Res",
    "Quasi-Ind", "Quasi-Res", "Neighborhood Comm", "Commerical", "Industrial",
])
FORMAS = metadatos.get("formas", [
    "Rectangular Shaped", "Irregular Shaped", "Trapezoidal Shaped", "Semi-rectangular Shaped",
])
DIRECCIONES = metadatos.get("direcciones", [
    "North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest",
    "No facing road",
])
TIPOS_CALLE = metadatos.get("tipos_calle", [
    "Ward Road", "City Road", "Prefectural Road", "Tokyo Metropolitan Road",
    "National Highway", "Private Road", "Agricultural Road",
])
TIPOS_OPERACION = [
    "Residential Land(Land and Building)",
    "Residential Land(Land Only)",
]


def construir_fila(entradas: dict, columnas_esperadas: list) -> pd.DataFrame:
    """Arma el DataFrame de una fila con TODAS las columnas que espera el pipeline.

    Las columnas que el formulario no cubre se completan con valores neutros:
    0 para numéricas y "Otros" para categóricas de alta cardinalidad. El pipeline
    tiene imputadores, así que tolera los faltantes.
    """
    fila = {col: np.nan for col in columnas_esperadas}
    fila.update({k: v for k, v in entradas.items() if k in columnas_esperadas})

    # Categóricas geográficas agrupadas: el usuario no las elige, se dejan en "Otros"
    for col in ("district_agrupado", "estacion_agrupada"):
        if col in fila and pd.isna(fila[col]):
            fila[col] = "Otros"

    return pd.DataFrame([fila])[columnas_esperadas]


def columnas_del_pipeline(artefacto) -> list:
    """Devuelve la lista de columnas en el orden que espera el ColumnTransformer."""
    return list(artefacto["cols_numericas"]) + list(artefacto["cols_categoricas"])


def tarjeta_metrica(titulo: str, valor: str, ayuda: str = "") -> str:
    """Devuelve el HTML de una tarjeta tipo métrica.

    Se usa en lugar de st.metric() porque ese componente carga un módulo JavaScript
    de forma dinámica, lo que falla en algunos navegadores/entornos. Este equivalente
    se renderiza con HTML/CSS puro y no tiene esa dependencia.
    """
    extra = f'<div style="font-size:0.75rem;opacity:0.55;margin-top:0.15rem;">{ayuda}</div>' if ayuda else ""
    return f"""
    <div style="border:1px solid rgba(128,128,128,0.25);border-radius:0.5rem;
                padding:0.85rem 1rem;margin-bottom:0.5rem;">
        <div style="font-size:0.8rem;opacity:0.7;">{titulo}</div>
        <div style="font-size:1.6rem;font-weight:650;line-height:1.3;">{valor}</div>
        {extra}
    </div>
    """


def formatear_yen(valor: float) -> str:
    """Formatea un monto en yenes de forma legible (¥ y separador de miles)."""
    if valor >= 1e8:
        return f"¥{valor:,.0f} (≈ {valor / 1e8:.2f} 億)"
    return f"¥{valor:,.0f}"


# --------------------------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------------------------
st.title("🏙️ Predicción inmobiliaria — Prefectura de Tokio")
st.caption(
    "Proyecto Integrador · IA y Aprendizaje Automático I · UCA · "
    "Datos: MLIT (Ministerio de Tierra, Infraestructura, Transporte y Turismo de Japón)"
)

tab_precio, tab_estructura, tab_info = st.tabs(
    ["💴 Estimar precio", "🏗️ Clasificar estructura", "ℹ️ Sobre los modelos"]
)


# ============================================================================================
# PESTAÑA 1 — Regresión: estimación del precio
# ============================================================================================
with tab_precio:
    st.subheader("Estimación del valor de transacción")
    st.markdown(
        "Completá las características del inmueble para obtener una estimación del "
        "**precio total de la operación**. El modelo fue entrenado sobre transacciones "
        "reales reportadas al MLIT entre 2005 y 2025."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ubicación**")
        p_ward = st.selectbox("Ward / municipio", WARDS, key="p_ward")
        p_zona = st.selectbox("Zona", ZONAS, key="p_zona")
        p_planning = st.selectbox("Zonificación urbana", CITY_PLANNING, key="p_planning")
        p_distancia = st.slider(
            "Distancia a la estación (minutos)", 0, 120, 10, key="p_dist",
            help="Tiempo a pie hasta la estación de tren más cercana.",
        )

    with col2:
        st.markdown("**Terreno y construcción**")
        p_tipo = st.selectbox("Tipo de operación", TIPOS_OPERACION, key="p_tipo")
        p_area_terreno = st.number_input(
            "Superficie del terreno (m²)", min_value=10.0, max_value=2000.0,
            value=120.0, step=10.0, key="p_area_t",
        )
        hay_edificio = p_tipo.endswith("(Land and Building)")
        p_area_construida = st.number_input(
            "Superficie construida (m²)", min_value=0.0, max_value=2000.0,
            value=100.0 if hay_edificio else 0.0, step=10.0,
            disabled=not hay_edificio, key="p_area_c",
            help="Solo aplica cuando la operación incluye edificio.",
        )
        p_anio_construccion = st.number_input(
            "Año de construcción", min_value=1900, max_value=2026,
            value=2000, step=1, disabled=not hay_edificio, key="p_anio_c",
        )

    with col3:
        st.markdown("**Parcela y entorno**")
        p_forma = st.selectbox("Forma del lote", FORMAS, key="p_forma")
        p_frente = st.number_input(
            "Frente del lote (m)", min_value=0.0, max_value=100.0,
            value=8.0, step=0.5, key="p_frente",
        )
        p_ancho_calle = st.number_input(
            "Ancho de la calle (m)", min_value=0.0, max_value=50.0,
            value=6.0, step=0.5, key="p_ancho",
        )
        p_cobertura = st.selectbox(
            "Building coverage ratio (%)", [30, 40, 50, 60, 70, 80, 100], index=3, key="p_cov",
        )
        p_far = st.selectbox(
            "Floor area ratio (%)", [50, 80, 100, 150, 200, 300, 400, 500, 600, 800],
            index=4, key="p_far",
        )

    with st.expander("Opciones avanzadas (uso del inmueble y operación)"):
        c1, c2 = st.columns(2)
        with c1:
            p_anio_op = st.number_input(
                "Año de la operación", min_value=2005, max_value=2026, value=2025, key="p_anio_op"
            )
            p_trimestre = st.selectbox("Trimestre", [1, 2, 3, 4], key="p_trim")
            p_direccion = st.selectbox("Orientación de la calle", DIRECCIONES, key="p_dir")
            p_tipo_calle = st.selectbox("Tipo de calle", TIPOS_CALLE, key="p_tcalle")
        with c2:
            p_uso_house = st.checkbox("Uso: vivienda", value=True, key="p_uh")
            p_uso_office = st.checkbox("Uso: oficina", key="p_uo")
            p_uso_shop = st.checkbox("Uso: comercio", key="p_us")
            p_uso_warehouse = st.checkbox("Uso: depósito", key="p_uw")
            p_uso_parking = st.checkbox("Uso: estacionamiento", key="p_up")
            p_uso_complex = st.checkbox("Uso: complejo habitacional", key="p_uc")

    st.divider()

    if st.button("Estimar precio", type="primary", use_container_width=True, key="btn_precio"):
        antiguedad = max(0, p_anio_op - p_anio_construccion) if hay_edificio else np.nan
        ratio_construido = (
            p_area_construida / p_area_terreno if hay_edificio and p_area_terreno > 0 else np.nan
        )

        entradas = {
            "ward": p_ward,
            "Area": p_zona,
            "City planning": p_planning,
            "Type": p_tipo,
            "Land : Shape": p_forma,
            "Frontage road : Direction": p_direccion,
            "Frontage road : Type": p_tipo_calle,
            "area_terreno_m2": p_area_terreno,
            "area_construida_m2": p_area_construida if hay_edificio else 0.0,
            "distancia_estacion_min": float(p_distancia),
            "distancia_estacion_nula": 0,
            "Frontage": p_frente,
            "Frontage road : Width": p_ancho_calle,
            "Building coverage ratio": float(p_cobertura),
            "Floor area ratio": float(p_far),
            "anio_operacion": p_anio_op,
            "trimestre_operacion": p_trimestre,
            "anio_construccion": float(p_anio_construccion) if hay_edificio else np.nan,
            "antiguedad": antiguedad,
            "antiguedad_inconsistente": 0,
            "antes_de_guerra": 1 if hay_edificio and p_anio_construccion < 1945 else 0,
            "ratio_construido": ratio_construido,
            "ratio_construido_sospechoso": 0,
            "area_terreno_capada": 1 if p_area_terreno >= 2000 else 0,
            "area_construida_capada": 1 if p_area_construida >= 2000 else 0,
            "sin_edificio": 0 if hay_edificio else 1,
            "tiene_nota_transaccion": 0,
            "uso_house": int(p_uso_house),
            "uso_office": int(p_uso_office),
            "uso_shop": int(p_uso_shop),
            "uso_warehouse": int(p_uso_warehouse),
            "uso_parking_lot": int(p_uso_parking),
            "uso_housing_complex": int(p_uso_complex),
        }

        cols = columnas_del_pipeline(regresor)
        fila = construir_fila(entradas, cols)

        # El modelo predice en escala log1p -> se deshace con expm1 para reportar en ¥.
        pred_log = float(regresor["pipeline"].predict(fila)[0])
        pred_yen = float(np.expm1(pred_log))

        # Intervalo aproximado: ±1 sigma en la escala logarítmica (donde el error es homocedástico),
        # luego transformado a yenes. Es asimétrico en ¥ por la naturaleza de la exponencial.
        lim_inf = float(np.expm1(pred_log - SIGMA_RESIDUOS_LOG))
        lim_sup = float(np.expm1(pred_log + SIGMA_RESIDUOS_LOG))

        st.markdown("### Resultado")
        c1, c2, c3 = st.columns(3)
        c1.markdown(tarjeta_metrica("Estimación central", formatear_yen(pred_yen)),
                    unsafe_allow_html=True)
        c2.markdown(tarjeta_metrica("Mínimo probable (≈68%)", formatear_yen(lim_inf)),
                    unsafe_allow_html=True)
        c3.markdown(tarjeta_metrica("Máximo probable (≈68%)", formatear_yen(lim_sup)),
                    unsafe_allow_html=True)

        precio_m2 = pred_yen / p_area_terreno if p_area_terreno > 0 else 0
        st.caption(f"Equivale a aproximadamente **¥{precio_m2:,.0f} por m²** de terreno.")

        # --- Comunicación honesta de la incertidumbre --------------------------------------
        st.info(
            "**Cómo leer este resultado.** El modelo se entrena sobre `log(precio)`, así que su "
            "error es proporcional, no absoluto: el rango mostrado corresponde a ±1 desvío en esa "
            "escala y por eso es asimétrico en yenes. En el conjunto de test el modelo alcanzó "
            "**R² = 0.85 en escala logarítmica**, pero un **MAPE del 140%**: es confiable para "
            "ubicar el orden de magnitud de una operación típica, no para tasar con precisión."
        )

        if pred_yen > UMBRAL_ALERTA_YEN:
            st.warning(
                "⚠️ **Estimación de baja confianza.** El valor cae en la cola alta de la "
                "distribución, donde el dataset tiene pocas operaciones y muy variables "
                "(edificios corporativos, grandes superficies). El error esperado en este rango "
                "es sustancialmente mayor que el promedio. Tomar el número como referencia "
                "orientativa, no como tasación."
            )

        if p_area_terreno >= 2000:
            st.warning(
                "La superficie de 2.000 m² es un **valor censurado** en la fuente original "
                "(el MLIT agrupa todo lo mayor bajo esa etiqueta), por lo que la estimación "
                "para lotes de ese tamaño es especialmente imprecisa."
            )


# ============================================================================================
# PESTAÑA 2 — Clasificación: estructura edilicia
# ============================================================================================
with tab_estructura:
    st.subheader("Clasificación de la estructura edilicia")
    st.markdown(
        "Estimá el **sistema constructivo** más probable de un edificio a partir de sus "
        "características. Clases: **W** (madera), **S** (acero), **RC** (hormigón armado), "
        "**SRC** (acero-hormigón) y **Otros**."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ubicación**")
        e_ward = st.selectbox("Ward / municipio", WARDS, key="e_ward")
        e_zona = st.selectbox("Zona", ZONAS, key="e_zona")
        e_planning = st.selectbox("Zonificación urbana", CITY_PLANNING, key="e_planning")
        e_distancia = st.slider("Distancia a la estación (min)", 0, 120, 10, key="e_dist")

    with col2:
        st.markdown("**Dimensiones**")
        e_area_terreno = st.number_input(
            "Superficie del terreno (m²)", min_value=10.0, max_value=2000.0,
            value=120.0, step=10.0, key="e_area_t",
        )
        e_area_construida = st.number_input(
            "Superficie construida (m²)", min_value=10.0, max_value=2000.0,
            value=100.0, step=10.0, key="e_area_c",
        )
        e_anio_construccion = st.number_input(
            "Año de construcción", min_value=1900, max_value=2026, value=2000, key="e_anio_c",
        )
        e_precio = st.number_input(
            "Precio de la operación (¥)", min_value=0, max_value=200_000_000_000,
            value=50_000_000, step=1_000_000, key="e_precio",
            help="El precio es un predictor válido acá, porque el target es la estructura.",
        )

    with col3:
        st.markdown("**Parcela y uso**")
        e_forma = st.selectbox("Forma del lote", FORMAS, key="e_forma")
        e_frente = st.number_input("Frente (m)", 0.0, 100.0, 8.0, 0.5, key="e_frente")
        e_ancho_calle = st.number_input("Ancho de calle (m)", 0.0, 50.0, 6.0, 0.5, key="e_ancho")
        e_cobertura = st.selectbox(
            "Building coverage ratio (%)", [30, 40, 50, 60, 70, 80, 100], index=3, key="e_cov"
        )
        e_far = st.selectbox(
            "Floor area ratio (%)", [50, 80, 100, 150, 200, 300, 400, 500, 600, 800],
            index=4, key="e_far",
        )

    with st.expander("Opciones avanzadas (uso del inmueble y operación)"):
        c1, c2 = st.columns(2)
        with c1:
            e_anio_op = st.number_input("Año de la operación", 2005, 2026, 2025, key="e_anio_op")
            e_trimestre = st.selectbox("Trimestre", [1, 2, 3, 4], key="e_trim")
            e_direccion = st.selectbox("Orientación de la calle", DIRECCIONES, key="e_dir")
            e_tipo_calle = st.selectbox("Tipo de calle", TIPOS_CALLE, key="e_tcalle")
        with c2:
            e_uso_house = st.checkbox("Uso: vivienda", value=True, key="e_uh")
            e_uso_office = st.checkbox("Uso: oficina", key="e_uo")
            e_uso_shop = st.checkbox("Uso: comercio", key="e_us")
            e_uso_warehouse = st.checkbox("Uso: depósito", key="e_uw")
            e_uso_parking = st.checkbox("Uso: estacionamiento", key="e_up")
            e_uso_complex = st.checkbox("Uso: complejo habitacional", key="e_uc")

    st.divider()

    if st.button("Clasificar estructura", type="primary", use_container_width=True, key="btn_estr"):
        antiguedad = max(0, e_anio_op - e_anio_construccion)
        ratio_construido = e_area_construida / e_area_terreno if e_area_terreno > 0 else np.nan

        entradas = {
            "ward": e_ward,
            "Area": e_zona,
            "City planning": e_planning,
            "Type": "Residential Land(Land and Building)",
            "Land : Shape": e_forma,
            "Frontage road : Direction": e_direccion,
            "Frontage road : Type": e_tipo_calle,
            "Total transaction value": float(e_precio),
            "precio_por_m2": e_precio / e_area_terreno if e_area_terreno > 0 else np.nan,
            "area_terreno_m2": e_area_terreno,
            "area_construida_m2": e_area_construida,
            "distancia_estacion_min": float(e_distancia),
            "distancia_estacion_nula": 0,
            "Frontage": e_frente,
            "Frontage road : Width": e_ancho_calle,
            "Building coverage ratio": float(e_cobertura),
            "Floor area ratio": float(e_far),
            "anio_operacion": e_anio_op,
            "trimestre_operacion": e_trimestre,
            "anio_construccion": float(e_anio_construccion),
            "antiguedad": antiguedad,
            "antiguedad_inconsistente": 0,
            "antes_de_guerra": 1 if e_anio_construccion < 1945 else 0,
            "ratio_construido": ratio_construido,
            "ratio_construido_sospechoso": 0,
            "area_terreno_capada": 1 if e_area_terreno >= 2000 else 0,
            "area_construida_capada": 1 if e_area_construida >= 2000 else 0,
            "sin_edificio": 0,
            "tiene_nota_transaccion": 0,
            "uso_house": int(e_uso_house),
            "uso_office": int(e_uso_office),
            "uso_shop": int(e_uso_shop),
            "uso_warehouse": int(e_uso_warehouse),
            "uso_parking_lot": int(e_uso_parking),
            "uso_housing_complex": int(e_uso_complex),
        }

        cols = columnas_del_pipeline(clasificador)
        fila = construir_fila(entradas, cols)

        pipeline = clasificador["pipeline"]
        pred_raw = pipeline.predict(fila)[0]
        if clasificador["usa_int"]:
            pred_clase = clasificador["int_a_clase"][int(pred_raw)]
        else:
            pred_clase = pred_raw

        probabilidades = pipeline.predict_proba(fila)[0]
        clases = clasificador["clases"]

        NOMBRES = {
            "W": "Madera (Wood)",
            "S": "Acero (Steel)",
            "RC": "Hormigón armado (Reinforced Concrete)",
            "SRC": "Acero-hormigón (Steel Reinforced Concrete)",
            "Otros": "Otros sistemas (LS, Block)",
        }

        st.markdown("### Resultado")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                tarjeta_metrica("Estructura más probable", str(pred_clase),
                                NOMBRES.get(pred_clase, "")),
                unsafe_allow_html=True)
            st.markdown(
                tarjeta_metrica("Confianza", f"{probabilidades.max() * 100:.1f}%"),
                unsafe_allow_html=True)

        with c2:
            df_prob = pd.DataFrame({
                "Estructura": [str(c) for c in clases],
                "Probabilidad": probabilidades,
            }).sort_values("Probabilidad", ascending=False)
            # Barras en HTML puro: st.bar_chart depende de Vega-Lite (módulo JS dinámico).
            barras = []
            for _, r in df_prob.iterrows():
                pct = float(r["Probabilidad"]) * 100
                barras.append(f"""
                <div style="margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                        <span><b>{r['Estructura']}</b></span><span>{pct:.1f}%</span>
                    </div>
                    <div style="background:rgba(128,128,128,0.18);border-radius:0.25rem;height:0.85rem;">
                        <div style="width:{max(pct, 0.5):.1f}%;background:#4C72B0;
                                    height:100%;border-radius:0.25rem;"></div>
                    </div>
                </div>""")
            st.markdown("".join(barras), unsafe_allow_html=True)

        st.dataframe(
            df_prob.assign(Probabilidad=lambda d: (d["Probabilidad"] * 100).round(2))
                   .rename(columns={"Probabilidad": "Probabilidad (%)"}),
            hide_index=True, use_container_width=True,
        )

        # --- Comunicación honesta de las limitaciones ---------------------------------------
        st.info(
            "**Cómo leer este resultado.** El dataset está fuertemente desbalanceado: la madera "
            "(`W`) representa el **80,8%** de los edificios, con una relación de **65,6 a 1** "
            "respecto de la clase más rara. El modelo alcanza **89,2% de accuracy** pero un "
            "**F1 macro de 0,566**: acierta muy bien en `W` y bastante peor en las minoritarias."
        )

        if pred_clase in ("SRC", "Otros"):
            st.warning(
                f"⚠️ **Clase minoritaria.** `{pred_clase}` es una de las categorías con menos "
                "ejemplos en el dataset, y el modelo tiene un recall bajo sobre ella "
                "(`Otros`: 0,05; `SRC`: 0,45). La predicción es menos confiable que para `W`."
            )

        if pred_clase in ("RC", "SRC"):
            st.caption(
                "Nota: `RC` y `SRC` son las clases que más se confunden entre sí, por tratarse "
                "de sistemas constructivos físicamente similares."
            )


# ============================================================================================
# PESTAÑA 3 — Información sobre los modelos
# ============================================================================================
with tab_info:
    st.subheader("Sobre los modelos desplegados")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        #### 💴 Regresor de precio (TP2)
        - **Algoritmo:** {regresor.get('modelo', 'XGBoost')}
        - **Target:** `Total transaction value` (¥), modelado sobre `log1p`
        - **Registros de entrenamiento:** 343.995 (toda la prefectura)
        - **R² (escala log):** 0,848 en test
        - **R² (escala ¥):** 0,267 en test
        - **RMSE:** ¥622.950.151 · **MAE:** ¥29.490.866
        - **MAPE:** 140,1%

        La diferencia entre el R² logarítmico y el R² en yenes se debe a que la
        transformación inversa (`expm1`) amplifica el error en la cola de precios altos.
        """)

    with c2:
        st.markdown(f"""
        #### 🏗️ Clasificador de estructura (TP3)
        - **Algoritmo:** {clasificador.get('modelo', 'XGBoost')}
        - **Target:** `estructura_principal` (W / S / RC / SRC / Otros)
        - **Registros de entrenamiento:** 224.622 (solo operaciones con edificio)
        - **Accuracy:** 89,2% en test
        - **F1 macro:** 0,566 · **F1 ponderado:** 0,876
        - **Kappa de Cohen:** 0,635
        - **ROC-AUC (one-vs-rest):** 0,948

        El desbalance es severo (65,6:1), por eso la métrica de referencia es el
        F1 macro y no la accuracy.
        """)

    st.divider()
    st.markdown("""
    #### Origen de los datos
    Los modelos se entrenaron sobre el registro público de **precios de transacciones
    inmobiliarias del MLIT** (Ministerio de Tierra, Infraestructura, Transporte y Turismo de
    Japón), acotado a la **Prefectura de Tokio** en el período **2005–2025**.

    #### Limitaciones que conviene tener presentes
    - Los datos son **autoreportados** por las partes de la operación, lo que puede introducir
      subreporte y redondeo en los montos.
    - La cobertura es exclusivamente de Tokio: **no generaliza** a otras prefecturas.
    - El registro **no incluye** atributos relevantes como calidad constructiva, estado de
      conservación o las condiciones particulares de cada negociación.
    - Las superficies mayores a 2.000 m² están **censuradas** en la fuente original.
    - Esta aplicación tiene fines **académicos y demostrativos**: no constituye una tasación
      profesional ni debe usarse como base para decisiones de inversión.
    """)

    st.divider()
    st.caption(
        "Proyecto Integrador · Inteligencia Artificial y Aprendizaje Automático I · 2026 · "
        "Licenciatura en Ciencia de Datos, UCA Rosario"
    )
