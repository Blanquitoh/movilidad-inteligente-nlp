# Movilidad Inteligente NLP

Proyecto modular para detectar y clasificar eventos de tráfico en tweets en español y recomendar intereses de usuario desde una única interfaz en Streamlit. Pensado para presentaciones académicas con énfasis en claridad, trazabilidad y configuración declarativa.

## Estructura del proyecto

```
├── configs/               # Configuración YAML centralizada
├── data/
│   ├── raw/               # Datasets originales (colocar manualmente)
│   └── processed/         # Datos limpiados y particionados
├── notebooks/             # Exploración de datos y visualizaciones
├── reports/               # Figuras generadas (p. ej. matriz de confusión)
├── scripts/               # ETL, entrenamiento y evaluación
├── src/
│   ├── core/              # Entidades de dominio
│   ├── infrastructure/    # Adaptadores NLP, geocodificación y recomendación
│   ├── interface/         # Aplicación Streamlit
│   └── use_cases/         # Casos de uso de aplicación
└── tests/                 # Pruebas unitarias e integrales
```

## Configuración

Todos los parámetros clave residen en `configs/config.yaml`:

- rutas de datos (raw, processed, artefactos de modelo)
- partición train/test (`test_size`, `random_state`)
- hiperparámetros del modelo (`max_features`, `C`, `n_neurons`)
- nivel de logging

## Flujo de trabajo

1. **Preparar datos**: descarga manualmente los datasets y colócalos en `data/raw/` con los nombres definidos en la configuración.
2. **ETL**: `python scripts/etl_build_dataset.py --config configs/config.yaml`
3. **Entrenamiento**: `python scripts/train_text_classifier.py --config configs/config.yaml`
4. **Evaluación**: `python scripts/evaluate_text_classifier.py --config configs/config.yaml`
5. **Interfaz web**: `streamlit run src/interface/web/app.py`

## Requisitos

- Python 3.10+
- Dependencias enumeradas en `pyproject.toml`
- Acceso opcional a Internet para geocodificación (Nominatim) y descarga inicial de stopwords de NLTK

## Pruebas

Ejecuta todas las pruebas con:

```
pytest
```

Las pruebas incluyen:
- limpieza de texto
- pipeline ETL + entrenamiento con un dataset sintético

## Licencia

MIT
