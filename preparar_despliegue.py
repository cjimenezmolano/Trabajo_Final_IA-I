"""
Prepara los artefactos para el despliegue de la app del TP4.

Qué hace:
  1. Copia los .joblib generados por el TP2 y el TP3 a `models/`.
  2. Extrae del dataset procesado los valores reales de cada variable categórica
     y los guarda en `models/metadatos.json`, para que los desplegables de la app
     ofrezcan exactamente las categorías que el modelo vio en entrenamiento.

Uso:
    python preparar_despliegue.py

Ejecutar desde la raíz del proyecto, después de haber corrido los notebooks TP1-TP3.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

# Rutas relativas a la raíz del proyecto
DIR_APP_MODELS = Path("models")
DIR_ORIGEN_MODELS = Path("outputs/models")
RUTA_DATASET = Path("outputs/processed/tokyo_tp1_procesado.csv")

MODELOS = ["tp2_mejor_regresor.joblib", "tp3_mejor_clasificador.joblib"]

# Columnas categóricas cuyas opciones se exponen en los formularios de la app
COLUMNAS_METADATOS = {
    "wards": "ward",
    "zonas": "Area",
    "city_planning": "City planning",
    "formas": "Land : Shape",
    "direcciones": "Frontage road : Direction",
    "tipos_calle": "Frontage road : Type",
}


def copiar_modelos() -> bool:
    """Copia los .joblib del pipeline de entrenamiento a la carpeta de la app."""
    DIR_APP_MODELS.mkdir(parents=True, exist_ok=True)
    ok = True

    for nombre in MODELOS:
        origen = DIR_ORIGEN_MODELS / nombre
        if origen.exists():
            shutil.copy2(origen, DIR_APP_MODELS / nombre)
            tamanio = (DIR_APP_MODELS / nombre).stat().st_size / 1e6
            print(f"  [ok] {nombre} ({tamanio:.1f} MB)")
        else:
            print(f"  [FALTA] {origen} — ejecutá el notebook correspondiente primero.")
            ok = False

    return ok


def exportar_metadatos() -> bool:
    """Extrae del dataset las categorías reales de cada variable del formulario."""
    if not RUTA_DATASET.exists():
        print(f"  [FALTA] {RUTA_DATASET} — ejecutá el TP1 primero.")
        return False

    # Solo se leen las columnas necesarias: el dataset completo son ~344k filas.
    columnas = list(COLUMNAS_METADATOS.values())
    df = pd.read_csv(RUTA_DATASET, usecols=columnas)

    metadatos = {}
    for clave, columna in COLUMNAS_METADATOS.items():
        # Ordenadas por frecuencia: las opciones más comunes aparecen primero en el desplegable.
        valores = df[columna].dropna().value_counts().index.tolist()
        metadatos[clave] = [str(v) for v in valores]
        print(f"  [ok] {clave}: {len(valores)} categorías")

    DIR_APP_MODELS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_APP_MODELS / "metadatos.json"
    ruta.write_text(json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [ok] metadatos.json guardado")
    return True


def main():
    print("Preparando artefactos para el despliegue\n")

    print("1. Copiando modelos serializados:")
    modelos_ok = copiar_modelos()

    print("\n2. Exportando metadatos del dataset:")
    metadatos_ok = exportar_metadatos()

    print()
    if modelos_ok and metadatos_ok:
        print("Listo. Ejecutá la app con:  streamlit run app.py")
    else:
        print("Faltan artefactos. Revisá los mensajes de arriba y volvé a ejecutar.")


if __name__ == "__main__":
    main()
