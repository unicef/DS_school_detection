import json
import re
from pathlib import Path, PosixPath

import geopandas as gpd
import mercantile
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box


def georef_tile(
    input_img: np.ndarray,
    input_filepath: PosixPath,
    width: int,
    height: int,
    tile_bbox: mercantile.LngLatBbox,
    output_folder: PosixPath,
    out_ext: str,
    img_type: str,
    dtype: str = "uint8",
):
    # Output filename
    filename_out = input_filepath.name.replace(
        f"{input_filepath.suffix}", f"_{img_type}.{out_ext}"
    )
    # Get number of bands
    bands = int(input_img.size / width / height)
    # Transformation to georeference the image
    transformation = rio.transform.from_bounds(
        tile_bbox.west,
        tile_bbox.south,
        tile_bbox.east,
        tile_bbox.north,
        width,
        height,
    )
    # Build metadata dict
    meta = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "dtype": dtype,
        "count": bands,
        "nodata": None,
        "crs": "EPSG:4326",
        "transform": transformation,
    }

    with rio.open(
        output_folder / filename_out,
        "w",
        **meta,
    ) as dst:
        for band in range(0, bands):
            dst.write(input_img[:, :, band], indexes=band + 1)
        dst.close()


def get_tile_bbox_from_fname(
    img_path: PosixPath,
):
    # Get tiles and zoom from filename
    xyz = re.findall(r"\d+", img_path.name)
    zoom, x_tile, y_tile = int(xyz[0]), int(xyz[1]), int(xyz[2])
    tile_xyz = mercantile.Tile(x=x_tile, y=y_tile, z=zoom)
    return mercantile.bounds(tile_xyz)


def shap_raster_sum_bands(input_folder: PosixPath, file_ext: str = "tiff"):
    for rfile in input_folder.glob(f"*.{file_ext}"):
        with rio.open(rfile, "r") as src:
            r, g, b = src.read()

        meta_copy = src.meta.copy()
        meta_copy["count"] = 1
        sv = np.stack([r, g, b]).transpose((1, 2, 0)).sum(-1)

        rfileout = rfile.name.replace(f".{file_ext}", f"_sum.{file_ext}")
        with rio.open(
            input_folder / rfileout,
            "w",
            **meta_copy,
        ) as dst:
            dst.write(sv, indexes=1)
            dst.close()
        print(f"File {str(input_folder / rfileout)} created")


def get_polygons_from_bbox(bbox):
    bbox_polygon = box(bbox.west, bbox.south, bbox.east, bbox.north)
    return bbox_polygon


def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame to rasterio compatible input format"""
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def extract_raster_data_from_bbox(raster, tile_bbox):
    mask_gdf = gpd.GeoDataFrame(
        {"geometry": tile_bbox}, index=[0], crs="EPSG:4326"
    )
    mask_gdf = mask_gdf.to_crs(crs=raster.crs.data)
    geojson_features = get_features(mask_gdf)
    samples, _ = mask(
        dataset=raster, shapes=geojson_features, all_touched=True, crop=True
    )
    return samples.flatten()


def get_samples_from_raster(raster_path: str, img_paths_array: np.ndarray) -> dict:
    samples = dict()
    raster = rio.open(raster_path, "r")

    for img_path in img_paths_array:
        # Get polygon bbox for each image
        f = Path(img_path[0])
        id_name = f.name.replace(f.suffix, "")
        bbox = get_tile_bbox_from_fname(f)
        bbox_poly = get_polygons_from_bbox(bbox)
        # Sample raster with polygon bbox
        sample = extract_raster_data_from_bbox(raster, bbox_poly)
        samples[id_name] = sample
    return samples

