import os
import json

import requests

from src.utils.load_config import load_config


class data_ingestor:
    def __init__(self):
        self.cfg = load_config()
        self.ROOT_PATH = self.cfg["paths"]["root"]
        self.RAW_PATH = self.cfg['paths']['raw_data']

    def ingest(self):
        for source in self.cfg['sources']:
            pth = self.ROOT_PATH + self.RAW_PATH + source
            os.makedirs(pth, exist_ok=True)
            payload = self._fetch_all(source)
            out_file = os.path.join(pth, f"{source}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"{source}: {len(payload['features'])} features -> {out_file}")

    def _fetch_all(self, source):
        """Page through an ArcGIS FeatureServer `query` endpoint (resultOffset/resultRecordCount)
        until the full result set has been retrieved, and return a single combined payload."""
        source_cfg = self.cfg['sources'][source]
        page_size = source_cfg['page_size']
        params = {**self.cfg['params'], "resultRecordCount": page_size}

        features = []
        meta = None
        offset = 0
        while True:
            r = requests.get(source_cfg['url'], params={**params, "resultOffset": offset}, timeout=60)
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
