# Salares Web — Anillo MESS (Huasco)

Sitio de difusión y mapa interactivo de vuelos de dron en el Salar de Huasco (objetivo específico 2, base preliminar).

## Estructura

```
salares_web/
├── index.html              # Inicio (contexto MESS + logos)
├── explorador.html         # Mapa Leaflet (RGB, índices, transparencia)
├── config.yaml
├── assets/img/             # Logos institucionales
├── data/
│   ├── drone/              # GeoTIFF planos para export (P7_YYYYMMDD_*.tif)
│   └── huasco/
│       ├── documents/      # Propuesta MESS (PDF)
│       ├── raster/P7/      # Ortomosaicos fuente
│       └── vector/         # KMZ + sitios_huasco.gpkg
├── data_static/            # WebP + JSON (generado)
└── scripts/
    ├── build_sitios_gpkg.py
    ├── prepare_drone_assets.py
    └── vis/                # export_data_ortho.py, pipeline_utils.py
```

El análisis científico (máscaras, PCA, PowerPoint) vive en el repositorio hermano **`salar_huasco`**.

## Uso

```powershell
cd e:\proyectos_github\salares_web
pip install -r requirements.txt

# Vectores (solo la primera vez, si no existe el GPKG)
python scripts/build_sitios_gpkg.py

# Recortar índices al polígono P7 (si cambian los TIFF en data/huasco/raster/P7)
python scripts/prepare_drone_assets.py

# Exportar capas para el navegador
python scripts/vis/export_data_ortho.py drone

python -m http.server 8090
# http://localhost:8090/  →  explorador.html
```

Por ahora solo **P7 (H3)** tiene capas listas; P1–P6 aparecen como polígonos de referencia.
