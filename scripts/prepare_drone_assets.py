"""
Genera GeoTIFF planos en ``data/drone/`` (P7_YYYYMMDD_índice.tif) desde
``data/huasco/raster/P7/``, recortados al polígono del sitio.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT

REPO = Path(__file__).resolve().parent.parent
RASTER_P7 = REPO / "data" / "huasco" / "raster" / "P7"
DRONE_DIR = REPO / "data" / "drone"
VECTORES = REPO / "data" / "huasco" / "vector" / "sitios_huasco.gpkg"
SITE_CODE = "P7"
FLIGHT_DATE = "20260129"


def _find_raster(*keywords: str) -> Path:
    for path in sorted(RASTER_P7.glob("*.tif")):
        name = path.name.lower()
        if all(k in name for k in keywords):
            return path
    raise FileNotFoundError(f"No se encontró TIFF con {keywords} en {RASTER_P7}")


def site_geometry():
    gdf = gpd.read_file(VECTORES)
    row = gdf.loc[gdf["codigo"].astype(str).str.upper() == SITE_CODE.upper()]
    if row.empty:
        raise ValueError(f"Sitio {SITE_CODE} no encontrado en {VECTORES}")
    return row.iloc[0].geometry


def geom_in_crs(geom, crs):
    gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs)
    return gs.__geo_interface__["features"][0]["geometry"]


def write_clipped(src_path: Path, out_path: Path, geom_wgs, band_indexes, dtype="float32"):
    with rasterio.open(src_path) as src:
        feat = geom_in_crs(geom_wgs, src.crs)
        data, transform = mask(
            src, [feat], crop=True, indexes=band_indexes, filled=True, nodata=np.nan if dtype == "float32" else 0
        )
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            count=len(band_indexes),
            dtype=dtype,
            compress="deflate",
            predictor=2,
        )
        if dtype == "float32":
            profile["nodata"] = np.nan
        else:
            profile.pop("nodata", None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(dtype, copy=False))


def write_index_from_rgb(rgb_path: Path, out_path: Path, geom_wgs, mode: str):
    with rasterio.open(rgb_path) as src:
        feat = geom_in_crs(geom_wgs, src.crs)
        bands, transform = mask(src, [feat], crop=True, indexes=[5, 6, 7], filled=True, nodata=0)
        red, rededge, nir = bands[0].astype(np.float64), bands[1].astype(np.float64), bands[2].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = (nir - red) / (nir + red) if mode == "ndvi" else (rededge - red) / (rededge + red)
        data[~np.isfinite(data)] = np.nan
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0], width=data.shape[1], transform=transform,
            count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=2,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)


def write_lst_aligned(thermal_path: Path, ref_path: Path, out_path: Path, geom_wgs):
    with rasterio.open(ref_path) as ref:
        feat = geom_in_crs(geom_wgs, ref.crs)
        with rasterio.open(thermal_path) as src:
            with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height, resampling=Resampling.bilinear) as vrt:
                data, transform = mask(vrt, [feat], crop=True, indexes=1, filled=True, nodata=np.nan)
        band = data[0] if getattr(data, "ndim", 0) == 3 else data
        profile = ref.profile.copy()
        profile.update(height=band.shape[0], width=band.shape[1], transform=transform, count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=2)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(band.astype(np.float32), 1)


def main() -> None:
    if not VECTORES.is_file():
        raise FileNotFoundError(f"Falta {VECTORES}. Ejecuta: python scripts/build_sitios_gpkg.py")

    geom = site_geometry()
    prefix = f"{SITE_CODE}_{FLIGHT_DATE}"
    DRONE_DIR.mkdir(parents=True, exist_ok=True)

    rgb_src = _find_raster("rgb")
    ndwi_src = _find_raster("ndwi")
    therm_src = _find_raster("thermal")

    print(f"RGB -> {prefix}_rgb.tif")
    write_clipped(rgb_src, DRONE_DIR / f"{prefix}_rgb.tif", geom, [1, 2, 3], dtype="uint16")

    for idx in ("ndvi", "ndci"):
        print(f"{idx.upper()} -> {prefix}_{idx}.tif")
        write_index_from_rgb(rgb_src, DRONE_DIR / f"{prefix}_{idx}.tif", geom, idx)

    print(f"NDWI -> {prefix}_ndwi.tif")
    write_clipped(ndwi_src, DRONE_DIR / f"{prefix}_ndwi.tif", geom, [1], dtype="float32")

    rgb_out = DRONE_DIR / f"{prefix}_rgb.tif"
    print(f"LST -> {prefix}_lst.tif")
    write_lst_aligned(therm_src, rgb_out, DRONE_DIR / f"{prefix}_lst.tif", geom)
    print(f"\nListo: {DRONE_DIR}")


if __name__ == "__main__":
    main()
