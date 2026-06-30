# ACD_JRC_TelemetriaVolvo

## Descripción

Análisis integral de datos de telemetría de la flota de 37 volquetes Volvo FMX de la empresa JRC, obtenidos mediante la API rFMS 2.1 de Volvo. El proyecto abarca el pipeline ETL completo, feature engineering orientado a KPIs operativos y un dashboard interactivo para la visualización de resultados.

- **Período analizado:** 20 de marzo al 27 de abril de 2026  
- **Dataset:** 111,821 registros crudos → 110,117 registros limpios (36 archivos JSONL)  
- **Empresa:** JRC (minería superficial — Perú, México y Canadá)  
- **Curso:** Análisis Computacional de Datos (DS3021) — UTEC, 2026-1

## Estructura del repositorio

```
ACD_JRC_TelemetriaVolvo/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_analitico.csv
│   └── sample_acumulado.csv
├── notebooks/
│   ├── limpieza.ipynb
│   └── feature_engineering_kpis.ipynb
├── dashboard/
│   └── app.py
└── docs/
    ├── diccionario_datos_inicial.pdf
    └── diccionario_datos_final.pdf
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar los notebooks en este orden:

1. `notebooks/limpieza.ipynb` — lee los 36 archivos JSONL crudos y genera:
   - `dataset_analitico.csv`
   - `dataset_acumulado.csv`

2. `notebooks/feature_engineering_kpis.ipynb` — lee los archivos anteriores y genera:
   - `dataset_master_kpis_analitico.csv`
   - `dataset_master_kpis_acumulado.csv`

Luego para el dashboard:

```bash
streamlit run dashboard/app.py
```

El dashboard consume `dataset_master_kpis_analitico.csv` y `dataset_master_kpis_acumulado.csv`. Colócalos dentro de la carpeta `data/` antes de ejecutar.

## Notebooks

- **limpieza.ipynb** — Lectura de 36 archivos JSONL, renombramiento de columnas, tratamiento de nulos, conversión de unidades y deduplicación.
- **feature_engineering_kpis.ipynb** — Cálculo de deltas por VIN, construcción de KPIs operativos, detección de outliers por IQR y exportación de datasets finales.

## Datos

Por restricciones de tamaño, la carpeta `data/` contiene muestras representativas de los datasets finales. Los datasets completos son:

- `dataset_master_kpis_analitico.csv` — 110,117 registros
- `dataset_master_kpis_acumulado.csv` — ~12,000 registros RFMS

## Integrantes

Grupo 4-6 — DS3021 Análisis Computacional de Datos

- Lazón Meza, María Fernanda
- Méndez Lázaro, Luis Fernando
- Llerena Silva, Nicolás Alejandro
- Nuñez Chiroque, Patricia Solange
- Timaná Castro, Aarón Victor Manuel
- Tipula Meza, Flavio Jose
- Wood de la fuente Chavez, Mia Alexie

**Profesor:** Espinoza Melgarejo, Jose Luis  
**UTEC — Lima, 2026**
