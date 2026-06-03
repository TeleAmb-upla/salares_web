"""
Salar de Huasco — exportación estática al estilo fic_agro / wetland_ortho_monitoring.
Exporta series, manifiestos y previsualizaciones raster por fuente (`sentinel2`, `drone`).

Dron: lee GeoTIFF bajo ``data/drone`` (y ``legacy_input_roots``), recursivamente. Con
``flat_date_filenames: true`` acepta ``G1_YYYYMMDD_ndvi.tif``, ``NOG_YYYYMMDD_rgb.tif``, etc.;
genera WebP en ``data_static/drone/rasters/`` y referencias en ``metadata.json`` para el
explorador (Leaflet ``imageOverlay``).
Los GeoTIFF flat se procesan uno tras otro; opcional ``pause_between_drone_tiffs_sec``.
Los índices (NDVI, etc.) pasan a WebP con ``vmin``/``vmax`` desde config tras remuestreo tipo RGB (sin cargar TIFF completo en RAM).
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pipeline_utils  # noqa: F401 — inicializa PROJ (proj.db) antes de rasterio/geopandas

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.mask import mask
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box, mapping, shape

from pipeline_utils import (
    build_file_fingerprint,
    ensure_master_aoi,
    ensure_predios_gv_utm19s_clone,
    get_source_input_roots,
    load_config,
    load_json_if_exists,
    resolve_years,
    write_json,
)

OUTPUT_DIR = Path("data_static")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Subir cuando cambien máscaras / recorte RGB en ``build_preview_rgb`` (invalida WebP cacheados con reuse).
RGB_PREVIEW_MASK_REVISION = 3

SEASON_MID_DATES = {
    "verano": "-02-15",
    "otono": "-05-15",
    "invierno": "-08-15",
    "primavera": "-11-15",
}

COLORMAPS = {
    "RdYlGn": [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 139),
        (255, 255, 191),
        (217, 239, 139),
        (166, 217, 106),
        (102, 189, 99),
        (26, 152, 80),
        (0, 104, 55),
    ],
    "RdYlBu": [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 144),
        (255, 255, 191),
        (224, 243, 248),
        (171, 217, 233),
        (116, 173, 209),
        (69, 117, 180),
        (49, 54, 149),
    ],
    "YlGn": [
        (255, 255, 204),
        (217, 239, 181),
        (173, 221, 142),
        (120, 198, 121),
        (65, 171, 93),
        (35, 139, 69),
        (0, 109, 44),
        (0, 68, 27),
    ],
    "BrBG": [
        (166, 97, 26),
        (223, 194, 125),
        (255, 255, 191),
        (199, 234, 229),
        (128, 205, 193),
        (53, 151, 143),
        (1, 102, 94),
        (0, 60, 48),
    ],
    "viridis": [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (53, 183, 121),
        (109, 205, 89),
        (180, 222, 44),
        (253, 231, 37),
    ],
    "inferno": [
        (0, 0, 4),
        (31, 12, 72),
        (85, 15, 109),
        (136, 34, 106),
        (186, 54, 85),
        (227, 89, 51),
        (249, 140, 10),
        (249, 201, 50),
        (252, 255, 164),
    ],
}


def leaflet_corners_from_affine_bounds(
    left: float,
    bottom: float,
    right: float,
    top: float,
    crs,
) -> list[list[float]] | None:
    """
    ``[[south, west], [north, east]]`` en EPSG:4326 para Leaflet.imageOverlay.fitBounds.

    Obligatorio proyectar cuando el ráster no está geográfico: si se guardaran metros proyectados,
    Leaflet interpreta valores como lat/lng y rompe overlay y zoom.
    """
    try:
        if crs is None:
            if max(abs(left), abs(right)) > 181.0 or max(abs(bottom), abs(top)) > 91.0:
                return None
            west, south, east, north = left, bottom, right, top
        else:
            west, south, east, north = transform_bounds(
                crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
            )
        return [[south, west], [north, east]]
    except Exception:
        return None


def tiff_footprint_leaflet_corners(tiff_path: str | Path) -> list[list[float]] | None:
    """Huella geográfica del GeoTIFF (extent nativo → WGS84) para previews y overlays."""
    path = Path(tiff_path)
    if not path.is_file():
        return None
    with rasterio.open(path) as src:
        return leaflet_corners_from_affine_bounds(
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, src.crs
        )


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_aoi(path: str | Path, id_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    if id_col not in gdf.columns:
        raise ValueError(f"Columna '{id_col}' no encontrada. Disponibles: {list(gdf.columns)}")
    return gdf


def preview_clip_geom_digest(geom_mapping: dict) -> str:
    """Huella de la geometría de recorte de previsualización (invalida WebP cache si cambia el AOI)."""
    return hashlib.sha256(shape(geom_mapping).wkb).hexdigest()[:32]


def json_dump(path: str | Path, payload: dict) -> None:
    write_json(path, payload)


def build_export_signature(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def reuse_existing_preview(
    existing_rasters: dict,
    raster_key: str,
    preview_path: str | Path,
    export_signature: str,
    source_fingerprint: dict,
    reuse_if_unchanged: bool = True,
) -> dict | None:
    if not reuse_if_unchanged:
        return None
    existing_entry = existing_rasters.get(raster_key)
    preview_path = Path(preview_path)
    if not existing_entry or existing_entry.get("export_signature") != export_signature:
        return None
    if existing_entry.get("source_fingerprint") != source_fingerprint:
        return None
    if not preview_path.exists():
        return None
    return existing_entry.get("visual")


def _reuse_previews_allowed(cfg: dict) -> bool:
    return bool(cfg.get("reuse_exported_webp_if_unchanged", True))


def _unlink_preview_output(preview_path: Path, cfg: dict) -> None:
    """Elimina la previsualización previa en disco antes de escribir (sustitución explícita)."""
    if not cfg.get("overwrite_preview_file", True):
        return
    try:
        preview_path.unlink(missing_ok=True)
    except OSError:
        pass


def build_existing_point_index(existing_timeseries: dict) -> dict:
    points_index = {}
    for wetland_id, wetland_entry in existing_timeseries.get("wetlands", {}).items():
        for index_key, index_entry in wetland_entry.get("indices", {}).items():
            for point in index_entry.get("points", []):
                period_key = point.get("period_key")
                if period_key:
                    points_index[(wetland_id, index_key, period_key)] = point
    return points_index


def colormap_legend_gradient(cmap_name: str) -> str:
    """CSS linear-gradient (abajo=vmin, arriba=vmax) alineado con ``apply_colormap``."""
    colors = COLORMAPS.get(cmap_name)
    if not colors:
        return "linear-gradient(to top, #888, #fff)"
    n = len(colors)
    parts = [
        f"rgb({int(r)},{int(g)},{int(b)}) {(i / (n - 1)) * 100:.1f}%"
        for i, (r, g, b) in enumerate(colors)
    ]
    return f"linear-gradient(to top, {', '.join(parts)})"


def apply_colormap(data: np.ndarray, nodata_mask: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    if cmap_name not in COLORMAPS:
        raise KeyError(f"Colormap no definido: {cmap_name}")
    colors = np.array(COLORMAPS[cmap_name], dtype=np.float64)
    nclr = len(colors)
    floats = np.asarray(data, dtype=np.float64)
    invisible = nodata_mask | ~np.isfinite(floats)

    norm = np.zeros_like(floats)
    vmin_f = float(vmin)
    vmax_f = float(vmax)
    denom = vmax_f - vmin_f + 1e-12
    ok = ~invisible
    if np.any(ok):
        norm[ok] = np.clip((floats[ok] - vmin_f) / denom, 0.0, 1.0)

    idx = norm * (nclr - 1)
    lo = np.clip(np.floor(idx).astype(np.int64, copy=False), 0, nclr - 1)
    hi = np.minimum(lo + 1, nclr - 1).astype(np.int64, copy=False)
    frac = (idx - lo)[..., np.newaxis]
    rgb = (colors[lo] * (1 - frac) + colors[hi] * frac).astype(np.uint8)
    rgba = np.zeros((*floats.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(invisible[..., np.newaxis], np.uint8(0), rgb)
    rgba[..., 3] = np.where(~invisible, np.uint8(255), np.uint8(0))
    return rgba


def _valid_reflectance_pixels(
    raster: np.ndarray,
    src,
    inside_mask: np.ndarray,
    n_bands: int,
) -> np.ndarray:
    """
    Píxeles donde cada banda usada es finita y no coincide con el nodata de esa banda.
    Evita que nodata/NaN pasen por el estiramiento y salgan como blanco en el WebP.
    """
    h, w = inside_mask.shape
    ok = np.ones((h, w), dtype=bool)
    cap = min(n_bands, int(raster.shape[0]))
    for ci in range(cap):
        bd = raster[ci].astype(np.float64, copy=False)
        ok &= np.isfinite(bd)
        nv = None
        nds = getattr(src, "nodatavals", None)
        if nds is not None and ci < len(nds):
            nv = nds[ci]
        if nv is None and ci == 0:
            nv = src.nodata
        if nv is not None and np.isfinite(float(nv)):
            ok &= ~np.isclose(bd, float(nv), rtol=1e-5, atol=1e-8)
    return ok & inside_mask


def _rgb_keep_non_border_fill(raster: np.ndarray, visualization_cfg: dict) -> np.ndarray:
    """
    Excluye píxeles típicos de relleno de mosaico (blanco/negro casi plano) cuando GDAL no marca nodata.
    También excluye bordes “pastel” (brillo alto y poca saturación): el umbral triple canal ~
    sólo atrapa blanco neutro y deja rosas/grises muy claros del marco TIFF.

    Ortos pueden venir como uint*/int o como float tras el pipe (p. ej. 0–255 en float): se infiere si el rango es
    ~[0,1] o fotográfico 8–16 bit para usar los umbrales adecuados.
    Devuelve máscara True = conservar píxel (misma forma H,W que las bandas).
    """
    if not visualization_cfg.get("rgb_hide_white_black_borders", True):
        return np.ones((raster.shape[1], raster.shape[2]), dtype=bool)
    r = raster[0].astype(np.float64, copy=False)
    g = raster[1].astype(np.float64, copy=False)
    b = raster[2].astype(np.float64, copy=False)
    finite = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    if not np.any(finite):
        return finite
    triple = np.stack([r, g, b], axis=0)
    flat_max = float(np.nanmax(triple[:, finite]))
    frac_w = float(visualization_cfg.get("rgb_border_white_frac_of_max", 0.982))
    frac_b = float(visualization_cfg.get("rgb_border_black_frac_of_max", 0.02))
    # Réflex ~[0,1] vs foto ~uint8 vs ~uint16
    if flat_max <= 1.25:
        wthresh = float(visualization_cfg.get("rgb_border_white_float_min", 0.98))
        bthresh = float(visualization_cfg.get("rgb_border_black_float_max", 0.02))
        near_white = (r >= wthresh) & (g >= wthresh) & (b >= wthresh)
        near_black = (r <= bthresh) & (g <= bthresh) & (b <= bthresh)
        spread_max = float(visualization_cfg.get("rgb_border_near_gray_spread_float", 0.14))
        mean_min = float(visualization_cfg.get("rgb_border_near_gray_mean_min_float", 0.88))
    else:
        vmax = 255.0 if flat_max <= 511.0 else 65535.0
        wthresh = vmax * frac_w
        bthresh = vmax * frac_b
        near_white = (r >= wthresh) & (g >= wthresh) & (b >= wthresh)
        near_black = (r <= bthresh) & (g <= bthresh) & (b <= bthresh)
        su8 = float(visualization_cfg.get("rgb_border_near_gray_spread_uint8", 42))
        mu8 = float(visualization_cfg.get("rgb_border_near_gray_mean_min_uint8", 226))
        scale = vmax / 255.0
        spread_max = su8 * scale
        mean_min = mu8 * scale
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    mean3 = (r + g + b) / 3.0
    near_pastel_fill = finite & (spread <= spread_max) & (mean3 >= mean_min)
    bad = near_white | near_black | near_pastel_fill
    return finite & ~bad


def _clip_geom_to_crs(geom_wgs84: dict, src_crs: rasterio.crs.CRS) -> dict:
    if not geom_wgs84 or str(src_crs) == "EPSG:4326":
        return geom_wgs84
    return transform_geom("EPSG:4326", src_crs, geom_wgs84)


def _metric_epsg_from_geometry(geom_wgs84) -> int:
    centroid = geom_wgs84.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = "south" if centroid.y < 0 else "north"
    return 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone


def buffer_geom_wgs84(geom_wgs84: dict, buffer_m: float) -> dict:
    if not geom_wgs84 or buffer_m <= 0:
        return geom_wgs84
    geom_shape = shape(geom_wgs84)
    buffered = (
        gpd.GeoSeries([geom_shape], crs="EPSG:4326")
        .to_crs(epsg=_metric_epsg_from_geometry(geom_shape))
        .buffer(buffer_m)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    return mapping(buffered)


def _clip_aoi_buffer_m(visualization_cfg: dict) -> float:
    """Metros de dilatación del AOI para previews (ventana/máscara WebP); 0 = polígono exacto."""
    raw = visualization_cfg.get("clip_aoi_buffer_m")
    if raw is not None:
        return max(0.0, float(raw))
    return max(0.0, float(visualization_cfg.get("rgb_context_buffer_m", 0)))


def _resolve_preview_shape(width: int, height: int, max_size: int | None) -> tuple[int, int]:
    if not max_size or max_size <= 0:
        return max(1, int(height)), max(1, int(width))
    scale = min(1.0, float(max_size) / float(max(width, height)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stretch_to_uint8(data: np.ndarray, valid_mask: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    band = np.asarray(data, dtype=np.float64, order="K")
    valid = band[valid_mask]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    lo, hi = np.percentile(valid, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.clip(np.nan_to_num(band, nan=0.0, posinf=255.0, neginf=0.0), 0, 255)
        return out.astype(np.uint8)

    stretched = np.clip((band - lo) / (hi - lo), 0, 1) * 255.0
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def _quantize_rgba(image: Image.Image, colors: int) -> Image.Image:
    if colors <= 0 or colors >= 256:
        return image
    rgb = image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    alpha = image.getchannel("A")
    return Image.merge("RGBA", (*rgb.split(), alpha))


def build_preview_rgb(
    raster_path: str | Path,
    preview_path: str | Path,
    visualization_cfg: dict,
    geom_wgs84: dict | None = None,
) -> dict:
    """
    Ortomosaico WebP recortado al polígono AOI (Gx UTM en WGS84).
    Usa primero ``rasterio.mask.mask(..., crop=True)`` para recortar por la forma del predio —
    evita mostrar todo el ráster cuando ``geometry_window`` equivale prácticamente al TIFF completo.
    """
    clip_to_aoi = visualization_cfg.get("clip_to_aoi", False) and geom_wgs84 is not None
    rgb_quality = int(visualization_cfg.get("rgb_webp_quality", 55))
    rgb_max_size = int(visualization_cfg.get("rgb_max_size", 8192))
    rgb_quantize_colors = int(visualization_cfg.get("rgb_quantize_colors", 64))
    stretch_percentiles = tuple(visualization_cfg.get("rgb_stretch_percentiles", [2, 98]))
    if len(stretch_percentiles) != 2:
        stretch_percentiles = (2, 98)

    leaflet_bbox = None
    native_h = native_w = None

    with rasterio.open(raster_path) as src:
        count = int(src.count)
        alpha_band = 4 if count >= 4 else None
        bands_to_read = [1, 2, 3] if count >= 3 else [1]

        clipped_ok = False
        stretch_arr_f = None
        raw_for_filters = None
        inside_mask = None

        if clip_to_aoi:
            geom_crs_clip = _clip_geom_to_crs(geom_wgs84, src.crs)
            try:
                clipped_ma, clip_tr = mask(
                    src,
                    [geom_crs_clip],
                    crop=True,
                    indexes=bands_to_read,
                    all_touched=False,
                    filled=False,
                )
                rgb_ma = np.ma.asarray(clipped_ma)
                nh_raw, nw_raw = int(rgb_ma.shape[1]), int(rgb_ma.shape[2])
                native_h, native_w = nh_raw, nw_raw

                # Máscara real del polígono según rasterio (evita desalineación respecto a geometry_mask al remuestrear).
                poly_ok_native = (~np.ma.getmaskarray(rgb_ma).any(axis=0)).astype(np.float32)

                planes_f: list[np.ndarray] = []
                planes_raw: list[np.ndarray] = []
                out_h, out_w = _resolve_preview_shape(nw_raw, nh_raw, rgb_max_size)
                same = out_h == nh_raw and out_w == nw_raw
                if same:
                    for bi in range(rgb_ma.shape[0]):
                        planes_f.append(
                            np.ma.filled(rgb_ma[bi].astype(np.float32, copy=False), np.nan)
                        )
                        planes_raw.append(
                            np.ma.filled(rgb_ma[bi].astype(np.float32, copy=False), np.nan)
                        )
                    stretch_arr_f = np.stack(planes_f, axis=0)
                    raw_for_filters = np.stack(planes_raw, axis=0)
                    work_tr = clip_tr
                    poly_ok_rs = poly_ok_native
                else:
                    l, b, r, tt = array_bounds(nh_raw, nw_raw, clip_tr)
                    dst_tr = from_bounds(l, b, r, tt, out_w, out_h)
                    work_tr = dst_tr
                    poly_ok_rs = np.empty((out_h, out_w), dtype=np.float32)
                    reproject(
                        source=poly_ok_native,
                        destination=poly_ok_rs,
                        src_transform=clip_tr,
                        src_crs=src.crs,
                        dst_transform=dst_tr,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest,
                    )
                    for bi in range(rgb_ma.shape[0]):
                        dest_f = np.empty((out_h, out_w), dtype=np.float32)
                        reproject(
                            source=np.ma.filled(rgb_ma[bi].astype(np.float32), np.nan),
                            destination=dest_f,
                            src_transform=clip_tr,
                            src_crs=src.crs,
                            dst_transform=dst_tr,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear,
                        )
                        dest_raw = np.empty((out_h, out_w), dtype=np.float32)
                        reproject(
                            source=np.ma.filled(rgb_ma[bi].astype(np.float32, copy=False), np.nan),
                            destination=dest_raw,
                            src_transform=clip_tr,
                            src_crs=src.crs,
                            dst_transform=dst_tr,
                            dst_crs=src.crs,
                            resampling=Resampling.nearest,
                        )
                        planes_f.append(dest_f)
                        planes_raw.append(dest_raw)
                    stretch_arr_f = np.stack(planes_f, axis=0)
                    raw_for_filters = np.stack(planes_raw, axis=0)

                nh2, nw2 = stretch_arr_f.shape[1], stretch_arr_f.shape[2]
                inside_mask = np.isfinite(stretch_arr_f).all(axis=0) & (poly_ok_rs > 0.5)

                left, bottom, right, top = array_bounds(nh2, nw2, work_tr)
                leaflet_bbox = leaflet_corners_from_affine_bounds(left, bottom, right, top, src.crs)
                clipped_ok = True
            except Exception as exc:
                print(f"    [rgb] rasterio.mask (clip AOI) falló ({raster_path}): {exc}")

        if not clipped_ok:
            geom_crs2 = None
            window = None
            native_window_w = int(src.width)
            native_window_h = int(src.height)
            bounds = src.bounds
            output_transform = src.transform
            native_h, native_w = native_window_h, native_window_w
            clip_on = clip_to_aoi
            if clip_on:
                geom_crs2 = _clip_geom_to_crs(geom_wgs84, src.crs)
                try:
                    window = geometry_window(src, [geom_crs2], pad_x=0, pad_y=0)
                    native_window_w = int(window.width)
                    native_window_h = int(window.height)
                    bounds = window_bounds(window, src.transform)
                    output_transform = src.window_transform(window)
                    native_h, native_w = native_window_h, native_window_w
                except Exception:
                    geom_crs2 = None
                    window = None
                    clip_on = False
                    bounds = src.bounds

            out_h, out_w = _resolve_preview_shape(native_window_w, native_window_h, rgb_max_size)
            read_kwargs = {
                "out_shape": (count, out_h, out_w),
                "resampling": Resampling.bilinear if count >= 3 else Resampling.nearest,
            }
            if window is not None:
                read_kwargs["window"] = window
            ras = src.read(**read_kwargs)
            stretch_arr_f = ras.astype(np.float32, copy=False)
            raw_for_filters = ras
            nh, nw = int(ras.shape[1]), int(ras.shape[2])
            if out_w != native_window_w or out_h != native_window_h:
                output_transform = output_transform * Affine.scale(native_window_w / out_w, native_window_h / out_h)

            inside_mask = np.ones((nh, nw), dtype=bool)
            if clip_on and geom_crs2 is not None:
                try:
                    inside_mask = geometry_mask(
                        [geom_crs2], out_shape=(nh, nw), transform=output_transform, invert=True
                    )
                except Exception:
                    inside_mask = np.ones((nh, nw), dtype=bool)

            left, bottom, right, top = (
                (bounds[0], bounds[1], bounds[2], bounds[3])
                if isinstance(bounds, (tuple, list)) and len(bounds) >= 4
                else (bounds.left, bounds.bottom, bounds.right, bounds.top)
            )
            leaflet_bbox = leaflet_corners_from_affine_bounds(left, bottom, right, top, src.crs)

        if leaflet_bbox is None:
            leaflet_bbox = tiff_footprint_leaflet_corners(raster_path)
        if leaflet_bbox is None:
            return None

        if count >= 3:
            stretch_ok = _valid_reflectance_pixels(raw_for_filters, src, inside_mask, 3)
            stretch_ok &= _rgb_keep_non_border_fill(raw_for_filters, visualization_cfg)
            r = _stretch_to_uint8(stretch_arr_f[0], stretch_ok, stretch_percentiles)
            g = _stretch_to_uint8(stretch_arr_f[1], stretch_ok, stretch_percentiles)
            b = _stretch_to_uint8(stretch_arr_f[2], stretch_ok, stretch_percentiles)
            r_u8 = np.where(stretch_ok, r, 0).astype(np.uint8)
            g_u8 = np.where(stretch_ok, g, 0).astype(np.uint8)
            b_u8 = np.where(stretch_ok, b, 0).astype(np.uint8)
            use_src_alpha = bool(visualization_cfg.get("rgb_use_source_alpha_band", False))
            if use_src_alpha and alpha_band is not None and stretch_arr_f.shape[0] >= alpha_band:
                ab_raw = raw_for_filters[alpha_band - 1]
                if np.issubdtype(ab_raw.dtype, np.integer) and np.iinfo(ab_raw.dtype).max > 255:
                    ab_f = ab_raw.astype(np.float64) / float(np.iinfo(ab_raw.dtype).max) * 255.0
                else:
                    ab_f = ab_raw.astype(np.float32)
                alpha_u8 = np.clip(ab_f, 0.0, 255.0).astype(np.uint8)
                alpha_u8 = np.where(stretch_ok, alpha_u8, np.uint8(0)).astype(np.uint8)
            else:
                alpha_u8 = np.where(stretch_ok, np.uint8(255), np.uint8(0)).astype(np.uint8)
            rgba = np.stack([r_u8, g_u8, b_u8, alpha_u8], axis=-1)
        else:
            stretch_ok = _valid_reflectance_pixels(raw_for_filters, src, inside_mask, 1)
            data = _stretch_to_uint8(stretch_arr_f[0], stretch_ok, stretch_percentiles)
            data_u8 = np.where(stretch_ok, data, 0).astype(np.uint8)
            alpha = np.where(stretch_ok, 255, 0).astype(np.uint8)
            rgba = np.stack([data_u8, data_u8, data_u8, alpha], axis=-1)

    image = Image.fromarray(rgba, "RGBA")
    image = _quantize_rgba(image, rgb_quantize_colors)
    upscale_factor = max(1, int(visualization_cfg.get("rgb_upscale_factor", visualization_cfg.get("upscale_factor", 1))))
    if upscale_factor > 1:
        image = image.resize(
            (image.width * upscale_factor, image.height * upscale_factor),
            resample=Image.Resampling.LANCZOS,
        )
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_format = str(visualization_cfg.get("preview_format", "WEBP")).upper()
    if preview_format == "WEBP":
        save_kwargs = {"quality": rgb_quality, "method": 6}
    else:
        save_kwargs = {"optimize": True}
    _unlink_preview_output(preview_path, visualization_cfg)
    image.save(preview_path, format=preview_format, **save_kwargs)
    return {
        "path": preview_path.relative_to(OUTPUT_DIR).as_posix(),
        "bounds": leaflet_bbox,
        "format": preview_format,
        "native_size": [native_w, native_h],
        "display_size": [int(image.width), int(image.height)],
        "render_mode": "smooth",
        "opacity": float(visualization_cfg.get("rgb_leaflet_opacity", 1.0)),
    }


def build_preview_raster(
    tiff_path: str | Path,
    preview_path: str | Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    visualization_cfg: dict,
    geom_wgs84: dict | None = None,
) -> dict | None:
    """
    WebP de índice (NDVI, etc.): remuestreo al vuelo como ``build_preview_rgb`` (``out_shape`` bilineal)
    para no leer el GeoTIFF completo; el color sigue **norma fija** ``vmin`` / ``vmax`` (``apply_colormap``),
    no estirado por percentiles tipo fotografía.
    """
    clip_to_aoi = visualization_cfg.get("clip_to_aoi", False) and geom_wgs84 is not None
    max_dim_cfg = visualization_cfg.get("index_preview_max_size") or visualization_cfg.get("rgb_max_size")

    leaflet_bbox = None

    with rasterio.open(tiff_path) as src:
        window = None
        geom_crs = None
        bounds_for_corners = src.bounds

        if clip_to_aoi:
            geom_crs = _clip_geom_to_crs(geom_wgs84, src.crs)
            try:
                window = geometry_window(src, [geom_crs], pad_x=0, pad_y=0)
                bounds_for_corners = window_bounds(window, src.transform)
            except Exception:
                clip_to_aoi = False
                window = None
                geom_crs = None
                bounds_for_corners = src.bounds

        nw = int(window.width) if window is not None else int(src.width)
        nh = int(window.height) if window is not None else int(src.height)
        out_h, out_w = _resolve_preview_shape(nw, nh, max_dim_cfg)

        if window is not None:
            ma = src.read(
                1,
                masked=True,
                window=window,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
            )
        else:
            ma = src.read(1, masked=True, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
        ma = np.ma.squeeze(ma)
        nodata_mask = np.ma.getmaskarray(ma).astype(bool)
        raw_for_norm = np.ma.filled(ma.astype(np.float64), np.nan)
        fill0 = np.where(np.isfinite(raw_for_norm), raw_for_norm, 0.0)
        path_name = Path(tiff_path).name.lower()
        if "_lst" in path_name:
            data = raw_for_norm.astype(np.float32)
        else:
            data = normalize_drone_index_band_values(fill0, src, 1).astype(np.float32)
        data[~np.isfinite(raw_for_norm)] = np.nan
        nodata_mask |= ~np.isfinite(data)

        wb = bounds_for_corners
        left, bottom, right, top = (
            (wb[0], wb[1], wb[2], wb[3])
            if isinstance(wb, (tuple, list)) and len(wb) >= 4
            else (wb.left, wb.bottom, wb.right, wb.top)
        )

        if clip_to_aoi and geom_crs is not None:
            out_transform = from_bounds(left, bottom, right, top, out_w, out_h)
            outside = ~geometry_mask([geom_crs], out_shape=(out_h, out_w), transform=out_transform, invert=True)
            nodata_mask |= outside

        leaflet_bbox = leaflet_corners_from_affine_bounds(left, bottom, right, top, src.crs)

    if leaflet_bbox is None:
        leaflet_bbox = tiff_footprint_leaflet_corners(tiff_path)
    if leaflet_bbox is None:
        return None

    rgba = apply_colormap(data, nodata_mask, vmin, vmax, cmap_name)

    image = Image.fromarray(rgba, "RGBA")
    upscale_factor = max(
        1, int(visualization_cfg.get("rgb_upscale_factor", visualization_cfg.get("upscale_factor", 1)))
    )
    if upscale_factor > 1:
        image = image.resize(
            (image.width * upscale_factor, image.height * upscale_factor),
            resample=Image.Resampling.LANCZOS,
        )

    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_format = str(visualization_cfg.get("preview_format", "WEBP")).upper()
    if preview_format == "WEBP":
        q = int(visualization_cfg.get("webp_quality_analytic", 86))
        save_kwargs = {"quality": max(30, min(q, 100)), "method": 6}
    else:
        save_kwargs = {"optimize": True}
    _unlink_preview_output(preview_path, visualization_cfg)
    image.save(preview_path, format=preview_format, **save_kwargs)
    return {
        "path": preview_path.relative_to(OUTPUT_DIR).as_posix(),
        "bounds": leaflet_bbox,
        "format": preview_format,
        "native_size": [nw, nh],
        "display_size": [int(image.width), int(image.height)],
        "render_mode": "smooth",
        "opacity": float(visualization_cfg.get("opacity", 0.9)),
    }


def _aoi_intersects_raster_extent_wgs84(src, geom_wgs84: dict) -> bool:
    """
    Comprueba si el AOI (WGS84) cruza el rectángulo del ráster en WGS84.
    Evita ``Input shapes do not overlap raster`` cuando el predio en config comparte un
    polígono demo pero el GeoTIFF está en otra ubicación.
    """
    try:
        b = src.bounds
        if src.crs:
            w, s, e, n = transform_bounds(
                src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
            )
        else:
            w, s, e, n = b.left, b.bottom, b.right, b.top
        footprint = box(w, s, e, n)
        aoi = shape(geom_wgs84)
        return bool(aoi.intersects(footprint))
    except Exception:
        return True


def normalize_drone_index_band_values(
    data: np.ndarray,
    src,
    band_idx: int = 1,
) -> np.ndarray:
    """
    Convierte la banda cruda de un GeoTIFF de índice (NDVI, NDWI, etc.) a magnitud física ~[-1, 1].

    Muchos flujos de fotogrametría multiespectral guardan el índice como entero 16 bits
    (valor × 10 000, p. ej. -2534 ↔ -0,2534). Sin reescalar, la media zonal y el colormap
    quedan incoherentes con una escala -1…1.

    Orden: aplica ``SCALE``/``OFFSET`` de GDAL si existen; luego, si el máximo absoluto
    sigue en (1500, 22000], divide entre 10 000 (evita uint8 NDVI ~0–255).
    """
    arr = np.asarray(data, dtype=np.float64)
    out = arr.copy()
    sc, off = 1.0, 0.0
    try:
        scales = getattr(src, "scales", None)
        offsets = getattr(src, "offsets", None)
        if scales and band_idx <= len(scales) and scales[band_idx - 1] is not None:
            sc = float(scales[band_idx - 1])
        if offsets and band_idx <= len(offsets) and offsets[band_idx - 1] is not None:
            off = float(offsets[band_idx - 1])
    except (TypeError, IndexError, ValueError):
        sc, off = 1.0, 0.0
    if sc != 1.0 or off != 0.0:
        out = out * sc + off

    finite = np.isfinite(out)
    if finite.any():
        mx = float(np.nanmax(np.abs(out[finite])))
        mn = float(np.nanmin(out[finite]))
        # Entorno ×10⁴ típico; valores bajos en AOI (p. ej. NDWI seco) pueden quedar <1500 sin reescalar.
        if 256.0 < mx <= 22000.0:
            out = out / 10000.0
        elif (
            mx > 1.5
            and mn >= -1e-6
            and mx <= 255.5
            and mx <= 2.5
            and mn <= 1.5
        ):
            # NDVI/NDWI guardados como byte 0–255 (no temperatura en °C ni índices ya físicos)
            out = (np.clip(out, 0.0, 255.0) / 255.0).astype(np.float64)
        elif mx > 1.5 and mn >= -128.5 and mx <= 127.5 and np.issubdtype(arr.dtype, np.signedinteger):
            out = (np.clip(out.astype(np.float64), -128.0, 127.0) / 128.0)
    return out.astype(np.float32)


def _trimmed_physical_mean(physical: np.ndarray, invalid: np.ndarray) -> float | None:
    """Media sobre píxeles válidos tras recorte [-1.5,1.5] y recorte 5–95% si hay muestra grande."""
    physical_ma = np.ma.array(physical, mask=invalid)
    v = np.ma.compressed(physical_ma)
    if v.size == 0:
        return None
    v = v.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    if v.size and np.nanmax(v) <= 60 and np.nanmin(v) >= -5:
        # Índices espectrales ~[-1, 1]
        v = v[(v >= -1.5) & (v <= 1.5)]
    if v.size == 0:
        return None
    if v.size >= 128:
        p05, p95 = np.percentile(v, [5.0, 95.0])
        trimmed = v[(v >= p05) & (v <= p95)]
        if trimmed.size >= 64:
            v = trimmed
    return float(np.mean(v))


def _extract_zonal_mean_gdalwarp(tiff_path: Path, zone_geom_wgs84: dict) -> float | None:
    """
    Algunos GeoTIFF tienen strips corruptos: rasterio.mask falla, pero gdalwarp a grillas
    pequeñas suele completar la lectura y basta para una media zonal representativa.
    """
    gdalwarp = shutil.which("gdalwarp")
    if not gdalwarp:
        return None
    warped = ""
    try:
        with rasterio.open(tiff_path) as src:
            if not _aoi_intersects_raster_extent_wgs84(src, zone_geom_wgs84):
                return None
            crs = src.crs
            zone_s = zone_geom_wgs84
            if crs and str(crs) != "EPSG:4326":
                zone_s = transform_geom("EPSG:4326", crs, zone_geom_wgs84)
            minx, miny, maxx, maxy = shape(zone_s).bounds
            nodata = src.nodata
        fd, warped = tempfile.mkstemp(suffix=".tif", prefix="fic_zonal_")
        os.close(fd)
        cmd = [gdalwarp, "-q", "-overwrite"]
        if nodata is not None and np.isfinite(float(nodata)):
            nd = float(nodata)
            nd_s = str(int(nd)) if nd.is_integer() else str(nd)
            cmd.extend(["-srcnodata", nd_s, "-dstnodata", nd_s])
        cmd.extend(
            [
                "-te",
                str(minx),
                str(miny),
                str(maxx),
                str(maxy),
                "-ts",
                "256",
                "256",
                "-r",
                "average",
                str(tiff_path.resolve()),
                warped,
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with rasterio.open(warped) as dst:
            raw = dst.read(1)
            invalid = np.zeros(raw.shape, dtype=bool)
            if dst.nodata is not None and np.isfinite(float(dst.nodata)):
                invalid |= np.isclose(
                    raw.astype(np.float64), float(dst.nodata), rtol=1e-5, atol=1e-6
                )
            invalid |= ~np.isfinite(raw.astype(np.float64))
            physical = normalize_drone_index_band_values(raw, dst, band_idx=1)
        return _trimmed_physical_mean(physical, invalid)
    except Exception as exc:
        print(f"    [zonal gdalwarp fallback] {exc}")
        return None
    finally:
        if warped:
            try:
                os.unlink(warped)
            except OSError:
                pass


def extract_zonal_mean(tiff_path: str | Path, zone_geom_wgs84: dict) -> float | None:
    """
    Media zonal dentro de ``zone_geom_wgs84``. Usa la misma normalización de escala que el WebP de índices.
    """
    tiff_path = Path(tiff_path)
    try:
        with rasterio.open(tiff_path) as src:
            if not _aoi_intersects_raster_extent_wgs84(src, zone_geom_wgs84):
                return None
            crs = src.crs
            zone_s = zone_geom_wgs84
            if crs and str(crs) != "EPSG:4326":
                zone_s = transform_geom("EPSG:4326", crs, zone_geom_wgs84)

            out_image, _ = mask(src, [zone_s], crop=True, indexes=1, nodata=src.nodata)
            raw = out_image[0]
            invalid = np.zeros(raw.shape, dtype=bool)
            if src.nodata is not None and np.isfinite(float(src.nodata)):
                invalid |= np.isclose(
                    raw.astype(np.float64), float(src.nodata), rtol=1e-5, atol=1e-6
                )
            invalid |= ~np.isfinite(raw.astype(np.float64))

            physical = normalize_drone_index_band_values(raw, src, band_idx=1)
            return _trimmed_physical_mean(physical, invalid)

    except Exception as exc:
        if "do not overlap" in str(exc).lower():
            return None
        if isinstance(exc, rasterio.RasterioIOError) or "Read failed" in str(exc):
            gv = _extract_zonal_mean_gdalwarp(tiff_path, zone_geom_wgs84)
            if gv is not None:
                print("    [zonal] Fallback gdalwarp tras error de lectura TIFF")
                return gv
        print(f"    Error: {exc}")
        return None


def compute_area_ha(gdf_wetland: gpd.GeoDataFrame) -> float:
    epsg = _metric_epsg_from_geometry(gdf_wetland.geometry.union_all())
    projected = gdf_wetland.to_crs(epsg=epsg)
    return round(float(projected.geometry.area.sum()) / 10_000, 1)


SEASON_ALIASES = {"otono": ["otono", "otoño"]}

def get_tiff_path(source_cfg: dict, wetland_id: str, year: int, season: str, index_key: str) -> Path | None:
    for root in get_source_input_roots(source_cfg):
        season_variants = SEASON_ALIASES.get(season, [season])
        exts = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG", ".jpg", ".JPG") if index_key == "rgb" else (".tif", ".tiff", ".TIF", ".TIFF")
        for s in season_variants:
            for ext in exts:
                candidate_new = root / wetland_id / index_key / f"{year}_{s}{ext}"
                if candidate_new.exists():
                    return candidate_new
                candidate_legacy = root / wetland_id / f"{year}_{s}_{index_key}{ext}"
                if candidate_legacy.exists():
                    return candidate_legacy
    return None


def _flat_filename_date_to_ymd(date_token: str) -> str:
    """Compacta el trozo de fecha del nombre de archivo a ``YYYYMMDD``."""
    digits = re.sub(r"\D", "", date_token.strip())
    if len(digits) < 8:
        raise ValueError(f"Fecha inválida en nombre de archivo: {date_token!r}")
    return digits[:8]


# P1…P7 (salar) o G1…G6 / NOG (fic) + fecha + índice.
FLAT_DRONE_NAME = re.compile(
    r"^((?:[Pp]\d+)|(?:[Gg]\d+)|(?:[Nn][Oo][Gg]))(?:[\s_]+)"
    r"((?:\d{8})|(?:\d{4}[_-]\d{2}[_-]\d{2}))(?:[\s_]+)"
    r"(?:([A-Za-z0-9]{1,12})_)?"
    r"(ndvi|ndwi|ndci|lst|rgb)\.(?:tif|tiff)$",
    re.I,
)


def discover_flat_drone_tiffs(source_cfg: dict) -> list[Path]:
    paths: list[Path] = []
    for root in get_source_input_roots(source_cfg):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
                continue
            if FLAT_DRONE_NAME.match(path.name):
                paths.append(path)
    return sorted(paths)


def _label_flat_period(ymd: str) -> tuple[str, str, int]:
    dt = datetime.strptime(ymd, "%Y%m%d").replace(tzinfo=timezone.utc)
    human = dt.strftime("%d %b %Y")
    return ymd, human, dt.year


def resolve_pause_between_drone_tiffs_sec(visualization_cfg: dict) -> float:
    """
    Segundos de espera entre un GeoTIFF y el siguiente (modo flat). El proceso ya es secuencial;
    esta pausa acota picos de I/O/RAM cuando los archivos son enormes.

    Sobrescribe la configuración si existe ``FIC_PAUSE_BETWEEN_DRONE_TIFFS_SEC``.
    """
    cfg_val = visualization_cfg.get("pause_between_drone_tiffs_sec")
    sec = 0.0
    if cfg_val is not None:
        try:
            sec = max(0.0, float(cfg_val))
        except (TypeError, ValueError):
            sec = 0.0
    env_raw = os.environ.get("FIC_PAUSE_BETWEEN_DRONE_TIFFS_SEC", "").strip()
    if env_raw:
        try:
            sec = max(0.0, float(env_raw))
        except ValueError:
            pass
    return sec


def ingest_flat_drone_date_assets(
    *,
    source_cfg: dict,
    source_key: str,
    visualization_cfg: dict,
    indices_cfg: dict,
    source_indices: list[str],
    wetland_ctx: dict[str, dict],
    rasters_dir: Path,
    existing_rasters: dict,
    periods_set: set,
    source_timeseries: dict,
    rasters_index: dict,
) -> int:
    paths = discover_flat_drone_tiffs(source_cfg)
    if not paths:
        return 0

    preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
    clip_aoi_buffer_m = _clip_aoi_buffer_m(visualization_cfg)
    # Recorte WebP y estadística: misma geometría (AOI maestro = predios seleccionados por predio).
    viz_flat = dict(visualization_cfg)
    n_new = 0
    reasons: dict[str, int] = {}
    first_skips: list[dict] = []
    pause_sec = resolve_pause_between_drone_tiffs_sec(visualization_cfg)
    if pause_sec > 0:
        print(
            "  [flat] Pausa entre GeoTIFF: "
            f"{pause_sec:g}s (`raster_visualization.pause_between_drone_tiffs_sec` o env "
            "`FIC_PAUSE_BETWEEN_DRONE_TIFFS_SEC`).",
            flush=True,
        )

    def bump_skip(reason: str, **extra: object) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        if len(first_skips) < 12:
            first_skips.append({"reason": reason, **{k: v for k, v in extra.items()}})

    for i, tiff_path in enumerate(paths):
        if i > 0 and pause_sec > 0:
            time.sleep(pause_sec)
        gc.collect()
        parsed = FLAT_DRONE_NAME.match(tiff_path.name)
        if not parsed:
            bump_skip("parse_failed", file=tiff_path.name)
            continue
        code_wid, ymd_raw, index_key = parsed.group(1).lower(), parsed.group(2), parsed.group(4).lower()
        try:
            ymd = _flat_filename_date_to_ymd(ymd_raw)
        except ValueError:
            bump_skip("invalid_date_token", file=tiff_path.name, ymd_raw=ymd_raw)
            continue
        if index_key not in source_indices:
            bump_skip("index_not_in_source", file=tiff_path.name, index_key=index_key)
            continue
        ctx = wetland_ctx.get(code_wid)
        if not ctx:
            print(f"  [flat] Sin AOI para predio {code_wid}: {tiff_path.name}")
            bump_skip("no_wetland_ctx", file=tiff_path.name, code_wid=code_wid)
            continue

        geom_union = ctx["geom_union"]
        preview_geom_flat = buffer_geom_wgs84(geom_union, clip_aoi_buffer_m)
        period_key, period_human, year = _label_flat_period(ymd)
        iso_date = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        raster_key = f"{code_wid}_{period_key}_{index_key}"
        if raster_key in rasters_index:
            bump_skip("duplicate_raster_key", file=tiff_path.name, raster_key=raster_key)
            continue

        stem = Path(tiff_path.name).stem
        preview_path = rasters_dir / f"{stem}.{preview_ext}"
        index_cfg = indices_cfg.get(index_key, {})
        visual_only = index_cfg.get("visual_only", False)
        source_fingerprint = build_file_fingerprint(tiff_path)

        if visual_only:
            export_signature = build_export_signature(
                {
                    "visualization": viz_flat,
                    "index": index_key,
                    "buffer_m": clip_aoi_buffer_m,
                    "preview_geom_sha": preview_clip_geom_digest(geom_union),
                    "mode": "rgb_visual_flat",
                    "rgb_preview_mask_revision": RGB_PREVIEW_MASK_REVISION,
                }
            )
            preview_meta = reuse_existing_preview(
                existing_rasters,
                raster_key,
                preview_path,
                export_signature,
                source_fingerprint,
                reuse_if_unchanged=_reuse_previews_allowed(viz_flat),
            )
            if preview_meta is None:
                preview_meta = build_preview_rgb(tiff_path, preview_path, viz_flat, preview_geom_flat)
        else:
            export_signature = build_export_signature(
                {
                    "visualization": viz_flat,
                    "index": index_key,
                    "colormap": index_cfg["colormap"],
                    "vmin": index_cfg["vmin"],
                    "vmax": index_cfg["vmax"],
                    "preview_geom_sha": preview_clip_geom_digest(geom_union),
                    "mode": "analytic_preview_flat",
                }
            )
            preview_meta = reuse_existing_preview(
                existing_rasters,
                raster_key,
                preview_path,
                export_signature,
                source_fingerprint,
                reuse_if_unchanged=_reuse_previews_allowed(viz_flat),
            )
            mean_value = extract_zonal_mean(tiff_path, geom_union)
            if preview_meta is None:
                preview_meta = build_preview_raster(
                    tiff_path,
                    preview_path,
                    index_cfg["colormap"],
                    index_cfg["vmin"],
                    index_cfg["vmax"],
                    viz_flat,
                    preview_geom_flat,
                )
            if preview_meta is None:
                print(f"  [flat] Sin datos válidos (sin vista previa ni media): {tiff_path.name}")
                bump_skip("zonal_and_preview_none", file=tiff_path.name, code_wid=code_wid)
                continue
            if mean_value is not None:
                point = {
                    "date": iso_date,
                    "label": period_human,
                    "year": year,
                    "season_key": "vuelo",
                    "season_label": period_human,
                    "period_key": period_key,
                    "value": round(mean_value, 4),
                }
                wentry = source_timeseries["wetlands"].setdefault(
                    code_wid,
                    {"name": ctx.get("name", code_wid), "indices": {}},
                )
                wentry["indices"].setdefault(index_key, {"points": [], "metrics": {}})
                wentry["indices"][index_key]["points"].append(point)

        if preview_meta is None:
            bump_skip("preview_meta_none_after_build", file=tiff_path.name, code_wid=code_wid)
            continue

        preview_meta = dict(preview_meta)
        if not viz_flat.get("clip_to_aoi", False):
            tiff_bounds_ll = tiff_footprint_leaflet_corners(tiff_path)
            if tiff_bounds_ll is not None:
                preview_meta["bounds"] = tiff_bounds_ll

        periods_set.add((period_key, f"Vuelo {period_human}"))
        rasters_index[raster_key] = {
            "source": source_key,
            "wetland_id": code_wid,
            "index": index_key,
            "year": year,
            "season": period_human,
            "period_key": period_key,
            "visual": preview_meta,
            "analytic_path": tiff_path.as_posix(),
            "source_fingerprint": source_fingerprint,
            "export_signature": export_signature,
        }
        n_new += 1
        print(f"  [flat] WebP <- {tiff_path.name} -> {preview_path.name}")

    return n_new


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return round(float(np.mean(arr)), 4)


def manifest_year_from_period_key(key: str | int | None) -> int | None:
    """Extrae año civil desde ``2024_verano`` o desde ``YYYYMMDD``."""
    if key is None:
        return None
    s = str(key).strip()
    if re.fullmatch(r"\d{8}", s):
        return int(s[:4])
    parts = s.split("_", 1)
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        return int(parts[0])
    return None


def compute_trend(points: list[dict]) -> dict | None:
    annual_map = {}
    for point in points:
        annual_map.setdefault(point["year"], []).append(point["value"])

    annual_summary = [
        {"year": year, "mean": round(float(np.mean(values)), 4)}
        for year, values in sorted(annual_map.items())
    ]
    if len(annual_summary) < 2:
        return None

    years = np.array([row["year"] for row in annual_summary], dtype=float)
    means = np.array([row["mean"] for row in annual_summary], dtype=float)
    slope, intercept = np.polyfit(years, means, 1)
    return {
        "annual_summary": annual_summary,
        "slope_per_year": round(float(slope), 4),
        "direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
        "line": [
            {"year": row["year"], "value": round(float((slope * row["year"]) + intercept), 4)}
            for row in annual_summary
        ],
    }


def compute_metrics(points: list[dict]) -> dict:
    if not points:
        return {
            "latest": None,
            "historical_mean": None,
            "seasonal_means": {},
            "annual_summary": [],
            "trend": None,
        }

    sorted_points = sorted(points, key=lambda item: item["date"])
    latest = sorted_points[-1]
    all_values = [point["value"] for point in sorted_points]
    historical_mean = mean_or_none(all_values)

    seasonal_groups = {}
    seasonal_labels = {}
    for point in sorted_points:
        seasonal_groups.setdefault(point["season_key"], []).append(point["value"])
        seasonal_labels.setdefault(point["season_key"], point["season_label"])
    seasonal_means = {
        key: {"label": seasonal_labels[key], "mean": mean_or_none(values)}
        for key, values in seasonal_groups.items()
    }

    seasonal_mean = seasonal_means.get(latest["season_key"], {}).get("mean")
    absolute_anomaly = round(latest["value"] - historical_mean, 4) if historical_mean is not None else None
    percent_anomaly = (
        round((absolute_anomaly / abs(historical_mean)) * 100, 2)
        if historical_mean not in (None, 0)
        else None
    )
    seasonal_abs = round(latest["value"] - seasonal_mean, 4) if seasonal_mean is not None else None
    seasonal_pct = (
        round((seasonal_abs / abs(seasonal_mean)) * 100, 2)
        if seasonal_mean not in (None, 0)
        else None
    )
    trend = compute_trend(sorted_points)

    latest_payload = {
        "date": latest["date"],
        "label": latest["label"],
        "year": latest["year"],
        "season_key": latest["season_key"],
        "season_label": latest["season_label"],
        "value": latest["value"],
        "historical_mean": historical_mean,
        "historical_anomaly_abs": absolute_anomaly,
        "historical_anomaly_pct": percent_anomaly,
        "seasonal_mean": seasonal_mean,
        "seasonal_anomaly_abs": seasonal_abs,
        "seasonal_anomaly_pct": seasonal_pct,
    }

    return {
        "latest": latest_payload,
        "historical_mean": historical_mean,
        "seasonal_means": seasonal_means,
        "annual_summary": trend["annual_summary"] if trend else [],
        "trend": trend,
    }


def export_source(
    config: dict,
    source_key: str,
    source_cfg: dict,
    gdf_aoi: gpd.GeoDataFrame,
    years: list[int],
) -> dict:
    id_col = config["shapefile_id_col"]
    indices_cfg = config["indices"]
    source_indices = source_cfg.get("indices") or list(indices_cfg.keys())
    visualization_cfg = config.get("raster_visualization", {})
    clip_aoi_buffer_m = _clip_aoi_buffer_m(visualization_cfg)
    source_output_dir = Path(source_cfg["static_root"])
    rasters_dir = source_output_dir / "rasters"
    csv_dir = source_output_dir / "csv"
    source_output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    existing_metadata_path = source_output_dir / "metadata.json"
    existing_timeseries_path = source_output_dir / "timeseries.json"
    existing_metadata = load_json_if_exists(existing_metadata_path, {})
    existing_timeseries = load_json_if_exists(existing_timeseries_path, {})
    existing_rasters = existing_metadata.get("rasters", {})
    existing_points = build_existing_point_index(existing_timeseries)

    source_timeseries = {
        "source": {
            "id": source_key,
            "label": source_cfg["label"],
            "description": source_cfg.get("description", ""),
            "has_data": False,
        },
        "wetlands": {},
    }
    rasters_index = {}
    wetland_ctx: dict[str, dict] = {}
    wetlands_info = {}
    periods_set = set()
    total_points = 0
    total_visual_rasters = 0

    print(f"\nFuente: {source_cfg['label']} ({source_key})")

    for wetland_id, wetland_cfg in config["wetlands"].items():
        wetland_name = wetland_cfg.get("name", wetland_id)
        gdf_w = gdf_aoi[gdf_aoi[id_col].astype(str).str.strip().str.lower() == wetland_id.lower()]
        if gdf_w.empty:
            print(f"  [WARN] Sin polígonos para {wetland_id}")
            wetlands_info[wetland_id] = {
                "name": wetland_name,
                "area_ha": None,
                "center": None,
                "n_periods": 0,
                "last_period": None,
                "available_years": [],
                "status": "missing_aoi",
            }
            continue

        area_ha = compute_area_ha(gdf_w)
        geom_union = mapping(gdf_w.geometry.union_all())
        preview_geom_union = buffer_geom_wgs84(geom_union, clip_aoi_buffer_m)
        center = gdf_w.geometry.union_all().centroid
        print(f"  - {wetland_name} ({wetland_id}) | {area_ha} ha")
        wetland_ctx[wetland_id] = {"name": wetland_name, "geom_union": geom_union}

        wetland_entry = {"name": wetland_name, "indices": {}}
        wetland_periods = set()

        for index_key in source_indices:
            index_cfg = indices_cfg.get(index_key, {})
            visual_only = index_cfg.get("visual_only", False)
            points = []

            for year in years:
                for season_key, season_label in config["seasons"].items():
                    tiff_path = get_tiff_path(source_cfg, wetland_id, year, season_key, index_key)
                    if tiff_path is None:
                        continue

                    if visual_only:
                        period_key = f"{year}_{season_key}"
                        label = f"{season_label} {year}"
                        wetland_periods.add((period_key, label))
                        periods_set.add((period_key, label))
                        raster_key = f"{wetland_id}_{period_key}_{index_key}"
                        preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
                        preview_name = f"{wetland_id}_{year}_{season_key}_{index_key}.{preview_ext}"
                        preview_path = rasters_dir / preview_name
                        source_fingerprint = build_file_fingerprint(tiff_path)
                        export_signature = build_export_signature(
                            {
                                "visualization": visualization_cfg,
                                "index": index_key,
                                "buffer_m": clip_aoi_buffer_m,
                                "preview_geom_sha": preview_clip_geom_digest(geom_union),
                                "mode": "rgb_visual",
                                "rgb_preview_mask_revision": RGB_PREVIEW_MASK_REVISION,
                            }
                        )
                        preview_meta = reuse_existing_preview(
                            existing_rasters,
                            raster_key,
                            preview_path,
                            export_signature,
                            source_fingerprint,
                            reuse_if_unchanged=_reuse_previews_allowed(visualization_cfg),
                        )
                        if preview_meta is None:
                            preview_meta = build_preview_rgb(tiff_path, preview_path, visualization_cfg, preview_geom_union)
                        rasters_index[raster_key] = {
                            "source": source_key,
                            "wetland_id": wetland_id,
                            "index": index_key,
                            "year": year,
                            "season": season_label,
                            "period_key": period_key,
                            "visual": preview_meta,
                            "analytic_path": Path(tiff_path).as_posix(),
                            "source_fingerprint": source_fingerprint,
                            "export_signature": export_signature,
                        }
                        total_visual_rasters += 1
                        continue

                    period_key = f"{year}_{season_key}"
                    label = f"{season_label} {year}"
                    raster_key = f"{wetland_id}_{period_key}_{index_key}"
                    preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
                    preview_name = f"{wetland_id}_{year}_{season_key}_{index_key}.{preview_ext}"
                    preview_path = rasters_dir / preview_name
                    source_fingerprint = build_file_fingerprint(tiff_path)
                    export_signature = build_export_signature(
                        {
                            "visualization": visualization_cfg,
                            "index": index_key,
                            "colormap": index_cfg["colormap"],
                            "vmin": index_cfg["vmin"],
                            "vmax": index_cfg["vmax"],
                            "preview_geom_sha": preview_clip_geom_digest(geom_union),
                            "mode": "analytic_preview",
                        }
                    )
                    preview_meta = reuse_existing_preview(
                        existing_rasters,
                        raster_key,
                        preview_path,
                        export_signature,
                        source_fingerprint,
                        reuse_if_unchanged=_reuse_previews_allowed(visualization_cfg),
                    )
                    cached_point = existing_points.get((wetland_id, index_key, period_key))
                    if preview_meta is not None and cached_point is not None:
                        point = dict(cached_point)
                    else:
                        mean_value = extract_zonal_mean(tiff_path, geom_union)
                        if mean_value is None:
                            continue
                        point = {
                            "date": f"{year}{SEASON_MID_DATES[season_key]}",
                            "label": label,
                            "year": year,
                            "season_key": season_key,
                            "season_label": season_label,
                            "period_key": period_key,
                            "value": round(mean_value, 4),
                        }
                    points.append(point)
                    wetland_periods.add((period_key, label))
                    periods_set.add((period_key, label))
                    if preview_meta is None:
                        preview_meta = build_preview_raster(
                            tiff_path,
                            preview_path,
                            index_cfg["colormap"],
                            index_cfg["vmin"],
                            index_cfg["vmax"],
                            visualization_cfg,
                            preview_geom_union,
                        )
                    rasters_index[raster_key] = {
                        "source": source_key,
                        "wetland_id": wetland_id,
                        "index": index_key,
                        "year": year,
                        "season": season_label,
                        "period_key": period_key,
                        "visual": preview_meta,
                        "analytic_path": Path(tiff_path).as_posix(),
                        "source_fingerprint": source_fingerprint,
                        "export_signature": export_signature,
                    }

            cleaned_points = [dict(point) for point in sorted(points, key=lambda item: item["date"])]
            metrics = compute_metrics(cleaned_points)
            wetland_entry["indices"][index_key] = {
                "points": cleaned_points,
                "metrics": metrics,
            }
            total_points += len(cleaned_points)

        source_timeseries["wetlands"][wetland_id] = wetland_entry
        latest_period = None
        if wetland_periods:
            latest_period = sorted(wetland_periods, key=lambda item: item[0])[-1][1]
        has_wetland_visuals = any(key.startswith(f"{wetland_id}_") for key in rasters_index)
        wetlands_info[wetland_id] = {
            "name": wetland_name,
            "area_ha": area_ha,
            "center": [round(center.y, 4), round(center.x, 4)],
            "n_periods": len(wetland_periods),
            "last_period": latest_period,
            "available_years": sorted({int(item[0].split("_")[0]) for item in wetland_periods}),
            "status": "ready" if (wetland_periods or has_wetland_visuals) else "empty",
        }

    if source_key == "drone" and source_cfg.get("flat_date_filenames", True):
        n_flat = ingest_flat_drone_date_assets(
            source_cfg=source_cfg,
            source_key=source_key,
            visualization_cfg=visualization_cfg,
            indices_cfg=indices_cfg,
            source_indices=source_indices,
            wetland_ctx=wetland_ctx,
            rasters_dir=rasters_dir,
            existing_rasters=existing_rasters,
            periods_set=periods_set,
            source_timeseries=source_timeseries,
            rasters_index=rasters_index,
        )
        total_visual_rasters += n_flat
        for wid, wentry in source_timeseries["wetlands"].items():
            for ix in wentry.get("indices", {}).values():
                pts = ix.get("points") or []
                cleaned = sorted(pts, key=lambda p: (str(p.get("date", "")), str(p.get("period_key", ""))))
                ix["points"] = [dict(x) for x in cleaned]
                ix["metrics"] = compute_metrics([dict(x) for x in cleaned])
            info = wetlands_info.get(wid)
            if not info or info.get("status") == "missing_aoi":
                continue
            period_meta = [(rv["period_key"], rv["season"]) for rv in rasters_index.values() if rv.get("wetland_id") == wid]
            uniq = {}
            for pk, lbl in period_meta:
                uniq[pk] = lbl
            if not uniq:
                continue
            pkeys_sorted = sorted(uniq.keys(), key=lambda pk: pk if isinstance(pk, str) else str(pk))
            last_lbl = uniq[pkeys_sorted[-1]]
            yrs = {rv["year"] for rv in rasters_index.values() if rv.get("wetland_id") == wid}
            wte = source_timeseries["wetlands"].get(wid, {})
            for ix in wte.get("indices", {}).values():
                for pt in ix.get("points", []):
                    y = pt.get("year")
                    if isinstance(y, int):
                        yrs.add(y)
            info["n_periods"] = len(uniq)
            info["last_period"] = last_lbl
            info["available_years"] = sorted(yrs)
            info["status"] = "ready" if uniq else info.get("status", "empty")

    source_has_data = total_points > 0 or total_visual_rasters > 0 or len(rasters_index) > 0
    source_timeseries["source"]["has_data"] = source_has_data

    for wetland_id, wetland_entry in source_timeseries["wetlands"].items():
        rows = []
        for index_key, index_entry in wetland_entry["indices"].items():
            for point in index_entry["points"]:
                rows.append(
                    {
                        "source": source_key,
                        "wetland_id": wetland_id,
                        "wetland_name": wetland_entry["name"],
                        "index": index_key.upper(),
                        "date": point["date"],
                        "year": point["year"],
                        "season": point["season_label"],
                        "period_key": point["period_key"],
                        "mean": point["value"],
                    }
                )
        if rows:
            csv_path = csv_dir / f"{wetland_id}_timeseries.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source",
                        "wetland_id",
                        "wetland_name",
                        "index",
                        "date",
                        "year",
                        "season",
                        "period_key",
                        "mean",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    sorted_periods = [{"key": key, "label": label} for key, label in sorted(periods_set, key=lambda item: item[0])]
    summary = {
        "n_wetlands": len(config["wetlands"]),
        "n_periods": len(sorted_periods),
        "data_points": total_points,
        "raster_count": len(rasters_index),
        "total_area_ha": round(
            sum(info["area_ha"] for info in wetlands_info.values() if info["area_ha"] is not None),
            1,
        ),
        "latest_period": sorted_periods[-1]["label"] if sorted_periods else None,
        "available_years": sorted(
            {
                y
                for item in sorted_periods
                if (y := manifest_year_from_period_key(item.get("key"))) is not None
            }
        ),
    }
    source_metadata = {
        "generated_at": iso_now(),
        "aoi_id_column": id_col,
        "source": {
            "id": source_key,
            "label": source_cfg["label"],
            "description": source_cfg.get("description", ""),
            "color": source_cfg.get("color"),
            "has_data": source_has_data,
        },
        "indices": {
            key: (
                {"label": value["label"], "description": value.get("description", ""), "visual_only": True}
                if value.get("visual_only")
                else {
                    "label": value["label"],
                    "description": value.get("description", ""),
                    "visual_only": False,
                    "vmin": value["vmin"],
                    "vmax": value["vmax"],
                    "colormap": value["colormap"],
                    "legend_gradient": colormap_legend_gradient(value["colormap"]),
                    "diverging": value.get("diverging", key != "lst"),
                }
            )
            for key, value in indices_cfg.items()
            if key in source_indices
        },
        "wetlands": wetlands_info,
        "rasters": rasters_index,
        "periods": sorted_periods,
        "summary": summary,
    }

    json_dump(source_output_dir / "timeseries.json", source_timeseries)
    json_dump(source_output_dir / "metadata.json", source_metadata)

    print(f"  Timeseries -> {(source_output_dir / 'timeseries.json').as_posix()}")
    print(f"  Metadata   -> {(source_output_dir / 'metadata.json').as_posix()}")

    rel_source_dir = source_output_dir.relative_to(OUTPUT_DIR).as_posix()
    return {
        "id": source_key,
        "label": source_cfg["label"],
        "description": source_cfg.get("description", ""),
        "color": source_cfg.get("color"),
        "has_data": source_has_data,
        "timeseries_path": f"{rel_source_dir}/timeseries.json",
        "metadata_path": f"{rel_source_dir}/metadata.json",
        "csv_dir": f"{rel_source_dir}/csv",
        "status": "ready" if source_has_data else "empty",
        "summary": summary,
    }


def export_static_data(selected_sources: list[str] | None = None) -> None:
    print("=" * 60)
    print("Exportación de datos – Salar de Huasco")
    print("=" * 60)

    config = load_config()
    ensure_predios_gv_utm19s_clone(REPO_ROOT)
    years = resolve_years(config)
    master_aoi_path = ensure_master_aoi(config)
    id_col = config["shapefile_id_col"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf_aoi = load_aoi(master_aoi_path, id_col)
    selected_ids = {key.lower() for key in config["wetlands"].keys()}
    gdf_aoi = gdf_aoi[gdf_aoi[id_col].astype(str).str.strip().str.lower().isin(selected_ids)].copy()
    print(f"AOIs cargados: {gdf_aoi[id_col].unique().tolist()}")

    aoi_output_path = OUTPUT_DIR / "wetlands_aoi.geojson"
    gdf_aoi.to_file(aoi_output_path, driver="GeoJSON")
    print(f"GeoJSON maestro -> {aoi_output_path.as_posix()}")

    existing_manifest = {}
    manifest_path = OUTPUT_DIR / "sources_manifest.json"
    if selected_sources and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)

    manifest = {
        "generated_at": iso_now(),
        "aoi_path": aoi_output_path.relative_to(OUTPUT_DIR).as_posix(),
        "year_range": {"start": years[0], "end": years[-1]},
        "sources": existing_manifest.get("sources", {}).copy() if selected_sources else {},
    }

    available_sources = {
        source_key: source_cfg
        for source_key, source_cfg in config.get("sources", {}).items()
        if source_cfg.get("enabled", True)
    }
    requested_sources = selected_sources or list(available_sources.keys())
    invalid_sources = [source for source in requested_sources if source not in available_sources]
    if invalid_sources:
        raise ValueError(f"Fuentes no válidas o deshabilitadas: {invalid_sources}")

    for source_key in requested_sources:
        source_cfg = available_sources[source_key]
        manifest["sources"][source_key] = export_source(config, source_key, source_cfg, gdf_aoi, years)

    json_dump(manifest_path, manifest)
    print(f"\nManifest -> {manifest_path.as_posix()}")
    print("\n--- Exportación finalizada ---")


if __name__ == "__main__":
    import sys

    export_static_data(sys.argv[1:] or None)
