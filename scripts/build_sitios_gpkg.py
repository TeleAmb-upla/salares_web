"""Construye ``data/vector/sitios_huasco.gpkg`` desde los KMZ de vuelo."""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parent.parent
KMZ_DIR = REPO / "data" / "vector" / "huasco_kml"
OUT = REPO / "data" / "vector" / "sitios_huasco.gpkg"


def main() -> None:
    os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")
    frames = []
    for kmz in sorted(KMZ_DIR.glob("HuascoP*.kmz")):
        code = kmz.stem.replace("Huasco", "").upper()
        gdf = gpd.read_file(kmz, on_invalid="fix")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
        gdf["codigo"] = code
        gdf["sitio"] = code
        gdf["es_h3"] = code == "P7"
        frames.append(gdf[["codigo", "sitio", "es_h3", "geometry"]])
    if not frames:
        raise FileNotFoundError(f"Sin KMZ en {KMZ_DIR}")
    import pandas as pd

    master = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_file(OUT, driver="GPKG")
    print(f"Escrito {OUT} ({len(master)} sitios)")


if __name__ == "__main__":
    main()
