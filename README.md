# Movilidad Inteligente NLP

Aplicación modular para detectar eventos de tráfico en publicaciones en español, priorizarlos en un mapa interactivo y recomendar temas a partir de intereses detectados automáticamente. El proyecto sigue principios de arquitectura limpia: las entidades viven en `src/core`, los casos de uso en `src/use_cases`, y las dependencias externas (NLP, datos, interfaces) se encapsulan en `src/infrastructure` y `src/interface`.

La interfaz en Streamlit ofrece dos flujos principales:

- **Detección de eventos**: clasifica tweets, geolocaliza incidentes cuando es posible y prioriza aquellos de mayor severidad.
- **Recomendaciones personalizadas**: identifica intereses del usuario, analiza sentimiento y sugiere temas relevantes desde un catálogo configurable.

## Estructura del proyecto

```
├── configs/               # Configuración declarativa (rutas de datos, parámetros, UI)
├── data/
│   ├── raw/               # Datasets originales (colocar manualmente)
│   └── processed/         # Datos intermedios persistidos por los scripts
├── notebooks/             # Exploración y reportes de EDA
├── reports/               # Artefactos generados (gráficas, métricas)
├── scripts/               # Comandos de ETL, entrenamiento y utilidades de bootstrap
├── src/
│   ├── core/              # Entidades del dominio de movilidad
│   ├── infrastructure/    # Adaptadores NLP, recomendadores y servicios externos
│   ├── interface/         # Interfaz web en Streamlit
│   └── use_cases/         # Casos de uso orquestan las entidades y adaptadores
└── tests/                 # Pruebas unitarias
```

## Configuración

El archivo `configs/config.yaml` centraliza los parámetros relevantes:

- rutas de artefactos (`paths.model_artifact`, `paths.recommendation_data`, `paths.sentiment_data`)
- partición de datos (`split.test_size`, `split.random_state`)
- hiperparámetros del modelo (`model.max_features`, `model.C`)
- segmentos demográficos para personalizar recomendaciones (`recommender.demographics.age_segments`)
- preferencias de la UI (`n_suggestions`, `top_keywords`) y credenciales opcionales (`maps.google_api_key`)

## Instalación rápida

1. Crear y activar un entorno virtual con Python 3.10+.
2. Instalar las dependencias: `pip install -e .`.
3. Descargar los datasets necesarios y ubicarlos en `data/raw/` según la configuración.

## Ejecución de la UI

Con las dependencias instaladas y los datos en su lugar, inicia la interfaz con:

```bash
streamlit run src/interface/web/app.py
```

La aplicación presenta dos pestañas:

- **Detección de eventos**: formulario para ingresar tweets, métricas de resumen, tabla de incidentes con enlaces a Google Maps y un mapa interactivo (requiere `maps.google_api_key`).
- **Recomendaciones personalizadas**: formulario para capturar publicaciones y demografía, tabla de intereses detectados, análisis de sentimiento y lista de sugerencias priorizadas.

## Flujo de scripts (opcional)

Para regenerar artefactos de entrenamiento:

1. **ETL**: `python scripts/etl_build_dataset.py --config configs/config.yaml`
2. **Entrenamiento**: `python scripts/train_text_classifier.py --config configs/config.yaml`
3. **Evaluación**: `python scripts/evaluate_text_classifier.py --config configs/config.yaml`

Estos comandos respetan la separación de responsabilidades: cada script sólo atiende una etapa del pipeline.

## Reportes de EDA automatizados

El proyecto incluye un comando dedicado para refrescar los artefactos analíticos almacenados en `notebooks/` y `reports/`. Los directorios deben existir previamente en el repositorio (incluyendo `reports/figures/` y `reports/metrics/`). Ejecuta:

```bash
python scripts/generate_eda_reports.py \
  --dataset data/processed/train.csv \
  --metrics-path reports/metrics/dataset_summary.json \
  --figures-dir reports/figures \
  --notebook-path notebooks/generated_eda_report.ipynb
```

El script produce:

- Un archivo JSON con métricas descriptivas (`reports/metrics/dataset_summary.json`).
- Gráficas de distribución en `reports/figures/` (categorías, longitud de texto y actividad temporal).
- Un notebook resumido en `notebooks/generated_eda_report.ipynb` que documenta hallazgos y muestra ejemplos de registros.

## Pruebas

Ejecuta la suite con:

```bash
pytest
```

Las pruebas cubren la segmentación de edades y la limpieza de texto, asegurando consistencia en los componentes reutilizables del recomendador y del pipeline NLP.

## Licencia

MIT
