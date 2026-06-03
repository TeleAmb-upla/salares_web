from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from shapely import make_valid
from shapely.geometry import box, mapping, shape as shp_shape


def _proj_dir_ready(root: Path) -> bool:
    return root.is_dir() and (root / "proj.db").is_file()


def bootstrap_proj_environment() -> None:
    """
    Rasterio/GDAL enlazan una lib PROJ concreta y exigen un ``proj.db`` compatible (layout reciente).

    En el venv suelen convivir bases distintas: ``pyproj/proj_dir/...`` (a veces vieja), ``/usr/share/proj``…
    Priorizamos **`rasterio/proj_data`**, que acompaña la rueda ``rasterio`` y suele coincidir con su lib.

    Si fallara, prueba: ``export PROJ_DATA=.../site-packages/rasterio/proj_data`` o reinstala deps.
    """

    def _apply_proj_dir(proj_root: Path) -> None:
        p = proj_root.resolve()
        os.environ["PROJ_DATA"] = str(p)
        os.environ.setdefault("PROJ_LIB", str(p))
        try:
            import pyproj.datadir as pdd

            pdd.set_data_dir(str(p))
        except Exception:
            pass

    if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
        first = (os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB") or "").split(os.pathsep)[0]
        if first:
            explicit = Path(first)
            if _proj_dir_ready(explicit):
                _apply_proj_dir(explicit)
                return

    ordered: list[Path] = []
    try:
        import rasterio

        ordered.append(Path(rasterio.__file__).resolve().parent / "proj_data")
    except Exception:
        pass
    try:
        import pyproj

        pj = Path(pyproj.__file__).resolve().parent
        ordered.extend((pj / "proj_dir" / "share" / "proj", pj / "proj_data" / "share" / "proj"))
    except Exception:
        pass
    try:
        import pyproj.datadir as _pdd

        ordered.append(Path(_pdd.get_data_dir()))
    except Exception:
        pass

    if os.environ.get("CONDA_PREFIX"):
        ordered.append(Path(os.environ["CONDA_PREFIX"]) / "share" / "proj")
    ordered.append(Path(sys.prefix) / "share" / "proj")

    ordered.extend((Path("/usr/share/proj"), Path("/usr/local/share/proj")))

    seen: set[str] = set()
    for cand in ordered:
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        if _proj_dir_ready(cand):
            _apply_proj_dir(cand)
            return


bootstrap_proj_environment()

import yaml

import geopandas as gpd
import pandas as pd
from shapely.ops import transform, unary_union

CONFIG_PATH = Path("config.yaml")


def _scalar_popup_text(val: object) -> str:
    """Texto seguro desde ``PopupInfo`` (evita NaN/float para regex)."""
    if pd.isna(val):
        return ""
    return str(val)


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json_if_exists(path: str | Path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_nan_for_json(obj):
    """
    Convierte NaN/Inf y escalares NumPy a valores JSON válidos (null donde el número no es finito).

    Por defecto ``json.dump`` puede escribir ``NaN`` (extensión), que ``JSON.parse`` del navegador rechaza.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_nan_for_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, int):
        return obj
    return obj


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_nan_for_json(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, ensure_ascii=False, indent=2, allow_nan=False)


def build_file_fingerprint(path: str | Path) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": file_path.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def resolve_years(config: dict) -> list[int]:
    years = config.get("years")
    if years:
        return [int(year) for year in years]

    start = int(config.get("year_start", datetime.now().year))
    end = config.get("year_end", datetime.now().year)
    if isinstance(end, str) and end.lower() == "auto":
        end = datetime.now().year
    end = int(end)
    if end < start:
        raise ValueError(f"Rango de años inválido: {start} > {end}")
    return list(range(start, end + 1))


def _gdf_from_geojson_file(path: Path) -> gpd.GeoDataFrame:
    """Construye un GeoDataFrame desde GeoJSON sin pasar por WKB de pyogrio (evita anillos KMZ/dron)."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features") or []
    rows = []
    for ft in feats:
        geom = shp_shape(ft["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        row = dict(ft.get("properties") or {})
        row["geometry"] = geom
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def read_wetland_aoi_dataset(aoi_source: str | Path, forced_crs: str | None = None) -> gpd.GeoDataFrame:
    """
    Lee un AOI desde .shp, .geojson, etc.
    Para .shp (p. ej. desde KMZ/DJI) usa ``ogr2ogr`` + JSON + shapely — ``gpd.read_file`` a veces falla
    en geometrías con anillos casi cerrados.
    """
    path = Path(aoi_source).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".shp":
        ogr = shutil.which("ogr2ogr")
        if ogr is None:
            os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")
            gdf = gpd.read_file(path)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".geojson", dir=str(path.parent))
            os.close(fd)
            os.unlink(tmp)
            tmp_p = Path(tmp)
            try:
                subprocess.run(
                    [
                        ogr,
                        "-overwrite",
                        "-f",
                        "GeoJSON",
                        "-t_srs",
                        "EPSG:4326",
                        str(tmp_p),
                        str(path),
                    ],
                    check=True,
                    env={**os.environ, "OGR_GEOMETRY_ACCEPT_UNCLOSED_RING": "YES"},
                    capture_output=True,
                    text=True,
                )
                gdf = _gdf_from_geojson_file(tmp_p)
            finally:
                tmp_p.unlink(missing_ok=True)
    else:
        os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")
        gdf = gpd.read_file(path)

    if forced_crs:
        gdf = gdf.set_crs(forced_crs, allow_override=True)
    return gdf


# Igual que ``export_data_ortho.FLAT_DRONE_NAME`` (GeoTIFF planos; separador _ o espacio).
_FLAT_DRONE_NAME = re.compile(
    r"^((?:[Gg]\d+)|(?:[Nn][Oo][Gg]))(?:[\s_]+)"
    r"((?:\d{8})|(?:\d{4}[_-]\d{2}[_-]\d{2}))(?:[\s_]+)"
    r"(ndvi|ndwi|rgb)\.(?:tif|tiff)$",
    re.I,
)


def _flat_drone_wetland_id_match(file_wetland_code: str, config_wetland_id: str) -> bool:
    return file_wetland_code.lower().strip() == config_wetland_id.lower().strip()


def list_flat_drone_tiffs_for_wetland(wetland_id: str, config: dict) -> list[Path]:
    """Todos los GeoTIFF planos del predio (ndvi / ndwi / rgb), rutas únicas."""
    drone_src = config.get("sources", {}).get("drone")
    if not drone_src:
        return []
    roots = get_source_input_roots(drone_src)
    by_kind: dict[str, list[Path]] = {"ndvi": [], "ndwi": [], "rgb": []}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
                continue
            m = _FLAT_DRONE_NAME.match(path.name)
            if not m or not _flat_drone_wetland_id_match(m.group(1), wetland_id):
                continue
            kind = m.group(3).lower()
            if kind not in by_kind:
                continue
            by_kind[kind].append(path)

    merged: list[Path] = []
    for kind in ("ndvi", "ndwi", "rgb"):
        merged.extend(by_kind[kind])

    seen: set[str] = set()
    out: list[Path] = []
    for path in merged:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    out.sort(key=lambda p: str(p.resolve()).lower())
    return out


def find_flat_drone_sample_tiff_for_wetland(wetland_id: str, config: dict) -> Path | None:
    """El GeoTIFF plano más reciente del predio (prioriza ndvi, luego ndwi, luego rgb)."""
    by_kind: dict[str, list[Path]] = {"ndvi": [], "ndwi": [], "rgb": []}
    for path in list_flat_drone_tiffs_for_wetland(wetland_id, config):
        m = _FLAT_DRONE_NAME.match(path.name)
        if not m:
            continue
        kind = m.group(3).lower()
        if kind in by_kind:
            by_kind[kind].append(path)
    for kind in ("ndvi", "ndwi", "rgb"):
        cand = by_kind[kind]
        if not cand:
            continue
        return max(cand, key=lambda p: p.stat().st_mtime_ns)
    return None


def _raster_footprint_box_wgs84(tiff_path: Path):
    """``shapely`` box en EPSG:4326 que cubre el GeoTIFF (densificado en borde proyectado)."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(tiff_path) as src:
        b = src.bounds
        crs = src.crs
        if crs:
            west, south, east, north = transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)
        else:
            west, south, east, north = b.left, b.bottom, b.right, b.top
    return box(west, south, east, north)


def _popup_numeric_ids_ordered(popupinfo_val: object) -> list[str]:
    """Números que aparecen como ``ID <n>`` en ``PopupInfo`` (orden aparición)."""
    txt = _scalar_popup_text(popupinfo_val)
    ids: list[str] = []
    for m in re.finditer(r"\bID\s+0*(\d+)\b", txt, flags=re.IGNORECASE):
        ids.append(str(int(m.group(1))))
    return ids


def _infer_plot_id_by_raster_anchor_point(
    gdf4326: gpd.GeoDataFrame,
    boxes: list,
    wetland_id: str,
    tiff_paths: list[Path],
) -> str | None:
    """
    Si las cajas WGS84 del orto no intersectan ningún polígono (solape de área = 0),
    sitúa el vuelo con un punto (centroide de la unión de cajas) y elige el ``ID`` del
    predio que lo contiene o el más cercano dentro de un umbral en metros.

    Cubre desalineaciones donde el ``plot_id`` histórico del shape no coincide con la
    huella del TIFF pero el centro del vuelo sí cae sobre el polígono correcto.
    """
    try:
        u = unary_union(boxes)
        pt = u.centroid
        if not u.covers(pt):
            pt = u.representative_point()
    except Exception:
        return None

    max_m = float(os.environ.get("FIC_DRONE_ANCHOR_MAX_DISTANCE_M", "12000"))

    try:
        gs_pt = gpd.GeoSeries([pt], crs="EPSG:4326")
        utm = gs_pt.estimate_utm_crs()
        if utm is None:
            utm = "EPSG:32719"
        pt_m = gs_pt.to_crs(utm).iloc[0]
        ser_m = gdf4326.geometry.to_crs(utm)
    except Exception:
        return None

    containing: list[tuple[str, float]] = []
    nearest: tuple[str, float] | None = None

    pops = (
        gdf4326["PopupInfo"]
        if "PopupInfo" in gdf4326.columns
        else pd.Series([None] * len(gdf4326), index=gdf4326.index)
    )
    for geom4326, geom_m, pop in zip(gdf4326.geometry, ser_m, pops):
        ids = _popup_numeric_ids_ordered(pop)
        if not ids:
            continue
        nid = ids[0]
        try:
            g2 = geom_m if geom_m.is_valid else make_valid(geom_m)
            if g2.covers(pt_m) or bool(g2.touches(pt_m)):
                g2d = geom4326 if geom4326.is_valid else make_valid(geom4326)
                containing.append((nid, float(g2d.area)))
                continue
            dist = float(g2.distance(pt_m))
            if nearest is None or dist < nearest[1]:
                nearest = (nid, dist)
        except Exception:
            continue

    if containing:
        containing.sort(key=lambda t: t[1])
        chosen = containing[0][0]
        print(
            f"  [aoi] {wetland_id}: plot_id {chosen!r} por punto del orto dentro del polígono (fallback; sin solape caja↔área).",
            flush=True,
        )
        return chosen

    if nearest is not None and nearest[1] <= max_m:
        print(
            f"  [aoi] {wetland_id}: plot_id {nearest[0]!r} por cercanía del centro del orto (~{nearest[1]:.0f} m al borde).",
            flush=True,
        )
        return nearest[0]

    return None


def infer_plot_id_from_drone_overlap(wetland_id: str, config: dict, gdf_unfiltered: gpd.GeoDataFrame) -> str | None:
    """
    Si hay GeoTIFF planos para ``wetland_id``, elige ``plot_id`` (``ID`` en ``PopupInfo``) del polígono
    que **maximiza la suma** de áreas de intersección con la huella WGS84 de **cada** ndvi/ndwi/rgb
    (corrige AOI cuando un solo índice no coincide con el shape pero el conjunto sí).
    """
    if os.environ.get("FIC_STRICT_CONFIG_PLOT_ID", "").strip():
        return None
    if "PopupInfo" not in gdf_unfiltered.columns:
        return None
    tiff_paths = list_flat_drone_tiffs_for_wetland(wetland_id, config)
    if not tiff_paths:
        return None
    boxes: list = []
    for p in tiff_paths:
        try:
            boxes.append(_raster_footprint_box_wgs84(p))
        except Exception as exc:
            print(f"  [WARN] Ortó huella omitida ({p.name}): {exc}", flush=True)

    if not boxes:
        return None

    gdf = gdf_unfiltered.copy()
    if gdf.crs is None:
        print(f"  [WARN] {wetland_id!r}: AOI sin CRS; no infiero plot_id por orto.")
        return None
    gdf4326 = gdf.to_crs("EPSG:4326")
    best_score = -1.0
    best_id: str | None = None
    for geom, pop in zip(gdf4326.geometry, gdf4326["PopupInfo"]):
        for nid in _popup_numeric_ids_ordered(pop):
            try:
                g2 = geom if geom.is_valid else make_valid(geom)
                score = sum(float(g2.intersection(b).area) for b in boxes)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_id = nid

    min_score_deg2 = 1e-10
    if best_id is None or best_score < min_score_deg2:
        anchor_id = _infer_plot_id_by_raster_anchor_point(gdf4326, boxes, wetland_id, tiff_paths)
        if anchor_id is not None:
            return anchor_id
        hint = ", ".join(p.name for p in tiff_paths[:5])
        if len(tiff_paths) > 5:
            hint += ", …"
        print(
            f"  [WARN] {wetland_id!r}: ningún polígono/PopupInfo corta las huellas de los ortó ({hint}); "
            f"mantengo plot_id de config.",
            flush=True,
        )
        return None

    return best_id


def wetland_preview_geom_wgs84(wetland_cfg: dict) -> dict | None:
    """
    Geometría en WGS84 (mapping GeoJSON) para **recortar** previsualizaciones de ráster.

    Si ``preview_aoi_source`` no está definido, devuelve ``None`` (el llamador usa la misma
    geometría que para estadística / AOI maestro).
    """
    preview_path = wetland_cfg.get("preview_aoi_source")
    if not preview_path:
        return None
    forced_crs = wetland_cfg.get("preview_aoi_crs") or wetland_cfg.get("aoi_crs")
    gdf = read_wetland_aoi_dataset(preview_path, forced_crs)
    if gdf.crs is None:
        raise ValueError(
            f"No hay CRS en {preview_path!r}. Define ``preview_aoi_crs: 'EPSG:32719'`` "
            f"en config para ese predio."
        )
    gdf = gdf.to_crs("EPSG:4326")
    if gdf.empty:
        raise ValueError(f"Geometría vacía en preview_aoi_source {preview_path!r}")
    return mapping(_to_2d(gdf.geometry.union_all()))


def ensure_predios_gv_utm19s_clone(repo_root: str | Path) -> Path | None:
    """
    Genera ``predios_gv_utm19s.*`` junto al shapefile fuente (WGS84 geográfico → EPSG:32719).
    Requiere ``ogr2ogr`` en PATH. No falla el llamador si falta fuente o herramienta (solo avisa).
    """
    root = Path(repo_root).resolve()
    src = root / "data/shapefiles/predios_gv/predios_gv.shp"
    dst = root / "data/shapefiles/predios_gv/predios_gv_utm19s.shp"
    if not src.is_file():
        print(f"  [WARN] No existe {src.as_posix()}; omito clon UTM.")
        return None
    ogr = shutil.which("ogr2ogr")
    if not ogr:
        print("  [WARN] ogr2ogr no encontrado; omito `predios_gv_utm19s.shp`. Instala GDAL.")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [ogr, "-overwrite", "-f", "ESRI Shapefile", "-t_srs", "EPSG:32719", str(dst), str(src)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  [WARN] ogr2ogr falló al generar UTM: {exc.stderr or exc.stdout or exc}")
        return None
    print(f"  Shapefile UTM 19S (WGS84) -> {dst.as_posix()}")
    return dst


def filter_gdf_by_kmz_popup_plot_id(gdf: gpd.GeoDataFrame, plot_id: str) -> gpd.GeoDataFrame:
    """
    Filtra filas del shape KMZ/export cuyo ``PopupInfo`` contiene una línea ``ID <n>`` típica
    de la capa ``predios_gv`` (espacios variables entre ``ID`` y el número).
    """
    col = "PopupInfo"
    if col not in gdf.columns:
        raise ValueError(
            f"plot_id definido pero el archivo no tiene la columna {col!r} "
            "(se espera en capas KMZ/export tipo ``predios_gv``)."
        )
    s = str(plot_id).strip()
    if not s.isdigit():
        raise ValueError(f'plot_id debe ser numérico (p. ej. "07"): {plot_id!r}')
    num = str(int(s))
    pat = re.compile(r"\bID\s+0*" + re.escape(num) + r"\b", re.IGNORECASE)
    mask = gdf[col].map(lambda t: bool(pat.search(_scalar_popup_text(t))))
    return gdf.loc[mask].copy()


def _wetland_layer_marker_normalized(wetland_id: str) -> str | None:
    """Ej. ``g6`` → ``G6``, ``nog`` → ``NOG``; ``lote_demo`` → ``None`` (no marca de capa)."""
    w = wetland_id.strip().lower()
    if w == "nog":
        return "NOG"
    m = re.fullmatch(r"g0*(\d+)", w)
    return f"G{int(m.group(1))}" if m else None


def _find_layer_column_name(gdf: gpd.GeoDataFrame) -> str | None:
    for c in gdf.columns:
        if isinstance(c, str) and c.lower() == "layer":
            return c
    return None


def _normalize_layer_attr_cell(cell: object) -> str:
    """Compara valores ``layer`` de KMZ tipo ``G6``, ``NOG``, ``predios``. SOG / G06 → canónico."""
    s = _scalar_popup_text(cell).strip().upper()
    if not s:
        return ""
    if re.match(r"^NOG\b", s):
        return "NOG"
    m = re.match(r"^G0*(\d+)(\b|$)", s)
    if m:
        return f"G{int(m.group(1))}"
    return s.split()[0][:12] if s else ""


def filter_gdf_by_site_code(gdf: gpd.GeoDataFrame, site_code: str) -> gpd.GeoDataFrame:
    """Filtra polígonos de sitios de vuelo (``codigo`` / ``sitio`` en GPKG del salar)."""
    target = str(site_code).strip().upper()
    for col in ("codigo", "code", "sitio", "site"):
        if col not in gdf.columns:
            continue
        norm = gdf[col].astype(str).str.strip().str.upper()
        hit = gdf.loc[norm == target]
        if len(hit):
            return hit.copy()
    return gdf.iloc[:0].copy()


def filter_gdf_by_layer_marker(gdf: gpd.GeoDataFrame, wetland_id: str) -> gpd.GeoDataFrame:
    """
    KMZ combinados pueden traer objetos extras con ``layer='G6'`` / ``'NOG'`` sin el mismo ``PopupInfo``
    que el resto de predios — priorizamos esas geometrías para el wetland homónimo.
    """
    tgt = _wetland_layer_marker_normalized(wetland_id)
    if tgt is None:
        return gdf.iloc[:0].copy()
    lcol = _find_layer_column_name(gdf)
    if not lcol:
        return gdf.iloc[:0].copy()
    norm = gdf[lcol].map(_normalize_layer_attr_cell)
    return gdf.loc[norm == tgt].copy()


def _to_2d(geometry):
    if geometry is None:
        return None
    return transform(lambda x, y, z=None: (x, y), geometry)


def ensure_master_aoi(config: dict) -> Path:
    output_path = Path(config["shapefile_path"])
    wetlands_cfg = config.get("wetlands", {})
    records = []

    os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")

    for wetland_id, wetland_cfg in wetlands_cfg.items():
        aoi_source = wetland_cfg.get("aoi_source")
        if not aoi_source:
            continue

        forced_crs = wetland_cfg.get("aoi_crs") or wetland_cfg.get("crs")
        gdf = read_wetland_aoi_dataset(aoi_source, forced_crs)
        site_code = wetland_cfg.get("site_code")
        if site_code:
            gdf = filter_gdf_by_site_code(gdf, site_code)
            if gdf.empty:
                raise ValueError(
                    f"Sin geometría para sitio {site_code!r} en {aoi_source!r} (columna codigo/sitio)."
                )
            print(f"  [aoi] {wetland_id!r}: polígono por site_code={site_code!r}.", flush=True)
        plot_id_cfg = wetland_cfg.get("plot_id")
        resolved_by_layer = False
        by_layer = filter_gdf_by_layer_marker(gdf, wetland_id)
        if len(by_layer) >= 1:
            if len(by_layer) > 1:
                print(
                    f"  [WARN] {wetland_id!r}: hay {len(by_layer)} filas con columna 'layer'; "
                    f"unión de geometrías para el AOI.",
                    flush=True,
                )
            gdf = by_layer
            resolved_by_layer = True
            print(
                f"  [aoi] {wetland_id!r}: polígono por columna 'layer' (objeto KMZ tipo G#/NOG, no sólo PopupInfo).",
                flush=True,
            )

        plot_id_eff: str | None = None
        inferred: str | None = None
        cfg_str = str(plot_id_cfg).strip() if plot_id_cfg is not None else ""
        if site_code:
            resolved_by_layer = True
        if not resolved_by_layer and plot_id_cfg is not None and cfg_str:
            inferred = infer_plot_id_from_drone_overlap(wetland_id, config, gdf)
            plot_id_eff = inferred if inferred is not None else cfg_str
            if inferred is not None and inferred != cfg_str:
                print(
                    f"  [aoi] {wetland_id}: plot_id en config ({cfg_str!r}) ≠ solape orto dron ({inferred!r}); "
                    f"uso este último.",
                    flush=True,
                )
            gdf = filter_gdf_by_kmz_popup_plot_id(gdf, plot_id_eff)

        if gdf.empty:
            if resolved_by_layer:
                msg = (
                    f"Sin geometría para predio {wetland_id!r} tras filtro por columna 'layer' "
                    f"en {aoi_source!r}."
                )
            elif plot_id_cfg is not None and cfg_str:
                msg = (
                    f"Sin geometría para predio {wetland_id!r}: revisa plot_id={plot_id_cfg!r} vs "
                    f"PopupInfo en {aoi_source!r}. Si hay GeoTIFF plano en data/drone, el pipeline "
                    f"puede infirir plot_id por orto (deshabilita con env FIC_STRICT_CONFIG_PLOT_ID=1)."
                )
            else:
                msg = f"Sin geometría después de leer {aoi_source!r}"
            raise ValueError(msg)

        if gdf.crs is None:
            raise ValueError(
                f"No hay CRS definido en {aoi_source}. Si tu geometría está en WGS84/UTM 19 Sur, "
                f'añade bajo ese predio en config.yaml por ejemplo:\n'
                f'  {wetland_id}:\n    aoi_source: "... "\n'
                '    aoi_crs: "EPSG:32719"'
            )

        gdf = gdf.to_crs("EPSG:4326")
        if gdf.empty:
            continue

        name_field = wetland_cfg.get("name_field")
        raw_name = None
        if name_field and name_field in gdf.columns:
            series = gdf[name_field].dropna().astype(str).str.strip()
            raw_name = series.iloc[0] if not series.empty else None

        records.append(
            {
                "wetland_id": wetland_id,
                "nombre": wetland_cfg.get("name") or raw_name or wetland_id,
                "fuente": str(aoi_source).replace("\\", "/"),
                "geometry": _to_2d(gdf.geometry.union_all()),
            }
        )

    if records:
        master_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        master_gdf.to_file(output_path, driver="GeoJSON")

    return output_path


def get_source_config(config: dict, source_key: str) -> dict:
    sources = config.get("sources", {})
    if source_key not in sources:
        raise KeyError(f"Fuente '{source_key}' no configurada")
    return sources[source_key]


def get_source_input_roots(source_cfg: dict) -> list[Path]:
    roots = [Path(source_cfg["input_root"])]
    roots.extend(Path(item) for item in source_cfg.get("legacy_input_roots", []))
    seen = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


def relative_posix(path: str | Path, start: str | Path = ".") -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()
