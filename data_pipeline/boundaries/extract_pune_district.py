"""Extract Pune district village polygons from the DataMeet Maharashtra boundary files.

Source: github.com/datameet/indian_village_boundaries, mh/mh1.geojson + mh/mh2.geojson.
Downloads are not committed to git; expected at data/raw_boundaries/mh{1,2}.geojson.
"""
import json
from pathlib import Path

import geopandas as gpd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_boundaries"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "boundaries"
OUT_PATH = OUT_DIR / "pune_villages.geojson"

DISTRICT_NAME = "PUNE"


def load_and_filter(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "DISTRICT" not in gdf.columns:
        raise ValueError(f"{path.name}: no DISTRICT column, found {list(gdf.columns)}")
    matches = gdf[gdf["DISTRICT"].astype(str).str.strip().str.upper() == DISTRICT_NAME]
    print(f"{path.name}: {len(gdf)} total features, {len(matches)} in {DISTRICT_NAME}")
    return matches


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for fname in ("mh1.geojson", "mh2.geojson"):
        fpath = RAW_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected {fpath} - download it first")
        parts.append(load_and_filter(fpath))

    combined = gpd.pd.concat(parts, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=parts[0].crs)

    if len(combined) == 0:
        raise RuntimeError(f"No villages found for DISTRICT == {DISTRICT_NAME!r} - check attribute values")

    print(f"\nCombined: {len(combined)} Pune district villages")
    print(f"CRS: {combined.crs}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Bounds (minx, miny, maxx, maxy): {combined.total_bounds}")

    sample = combined.iloc[0]
    print(f"\nSample feature: NAME={sample.get('NAME')}, SUB_DIST={sample.get('SUB_DIST')}, TYPE={sample.get('TYPE')}")

    combined.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
