import os
import io
import json
import subprocess
import zipfile
import requests
from pathlib import Path

import pandas as pd
from src.utils.load_config import load_config


class data_ingestor:
    def __init__(self):
        self.cfg = load_config()
        self.ROOT_PATH = self.cfg["paths"]["root"]
        self.RAW_PATH = self.cfg['paths']['raw_data']

    def ingest(self):
        """Download the Overture places file, the Mecklenburg GIS layers, the NC school
        report cards and the Zillow index (each skipped if already present), then page
        through each ArcGIS source in config `sources` (none are active while the
        project is Mecklenburg-only - that county's CSV is placed by hand)."""
        nc_places = self.ROOT_PATH + self.RAW_PATH + 'NC parquet/nc_places.parquet'
        self.fetch_places(nc_places, self.cfg['nc_places']['bbox'])
        self.fetch_meck_layers()
        self.fetch_report_cards()
        self.fetch_zillow()
        for source in self.cfg.get('sources') or {}:
            pth = self.ROOT_PATH + self.RAW_PATH + source
            os.makedirs(pth, exist_ok=True)
            source_cfg = self.cfg['sources'][source]
            payload = self._fetch_all(source_cfg['url'], source_cfg['page_size'], self.cfg['params'])
            out_file = os.path.join(pth, f"{source}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"{source}: {len(payload['features'])} features -> {out_file}")

    def fetch_places(self,_,bbox):
        _ = Path(_)
        if _.exists():
            return _
        _.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "overturemaps", "download",
            "--type=place",
            f"--bbox={bbox}",
            "-f", "geoparquet",
            "-o", str(_),
        ], check=True)
        return _

    def fetch_meck_layers(self):
        """Each layer in config `meck_gis.layers` as GeoJSON (WGS84), plus the chosen
        Quality of Life indicators for one year as a csv (attributes only - the NPA
        polygons are the NeighborhoodProfileAreas layer)."""
        gis = self.cfg['meck_gis']
        pth = Path(self.ROOT_PATH + self.RAW_PATH + 'meck_gis')
        pth.mkdir(parents=True, exist_ok=True)
        for layer in gis['layers']:
            out_file = pth / f"{layer}.geojson"
            if out_file.exists():
                continue
            params = {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"}
            payload = self._fetch_all(gis['url'].format(layer=layer), gis['page_size'], params)
            out_file.write_text(json.dumps(payload))
            print(f"{layer}: {len(payload['features'])} features -> {out_file}")

        qol = gis['quality_of_life']
        out_file = pth / "quality_of_life.csv"
        if not out_file.exists():
            names = ",".join(f"'{name}'" for name in qol['indicators'])
            params = {
                "where": f"raw_data_name in ({names}) and year = '{qol['year']}'",
                "outFields": "npa,raw_data_name,year,raw,normalized,normalized_data_units",
                "returnGeometry": "false", "f": "json",
            }
            payload = self._fetch_all(gis['url'].format(layer=qol['layer']), gis['page_size'], params)
            pd.DataFrame([f["attributes"] for f in payload["features"]]).to_csv(out_file, index=False)
            print(f"quality of life: {len(payload['features'])} rows -> {out_file}")

    def fetch_report_cards(self):
        """Download the NC School Report Card zips and extract config `report_cards.files`."""
        rc = self.cfg['report_cards']
        pth = Path(self.ROOT_PATH + self.RAW_PATH + 'school_report_cards')
        if all((pth / f).exists() for f in rc['files']):
            return
        pth.mkdir(parents=True, exist_ok=True)
        for url in rc['urls']:
            # dpi.nc.gov rejects requests without a browser-like User-Agent
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for name in z.namelist():
                    if os.path.basename(name) in rc['files']:
                        (pth / os.path.basename(name)).write_bytes(z.read(name))
                        print(f"{os.path.basename(name)} -> {pth}")

    def fetch_zillow(self):
        out_file = Path(self.ROOT_PATH + self.RAW_PATH + 'zillow/county_zhvi.csv')
        if out_file.exists():
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(self.cfg['zillow_zhvi'], timeout=300)
        r.raise_for_status()
        out_file.write_bytes(r.content)
        print(f"zillow county index -> {out_file}")

    def _fetch_all(self, url, page_size, params):
        """Page through an ArcGIS FeatureServer `query` endpoint (resultOffset/resultRecordCount)
        until the full result set has been retrieved, and return a single combined payload."""
        params = {**params, "resultRecordCount": page_size}
        features = []
        meta = None
        offset = 0

        while True:
            r = requests.get(url, params={**params, "resultOffset": offset}, timeout=60)
            r.raise_for_status()
            page = r.json()
            if meta is None:
                meta = {k: v for k, v in page.items() if k != "features"}
            batch = page.get("features", [])
            features.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)

        meta["features"] = features
        meta["exceededTransferLimit"] = False

        return meta

    def health_check(self, response):
        return response.status_code == 200


if __name__ == "__main__":
    data_ingestor().ingest()
