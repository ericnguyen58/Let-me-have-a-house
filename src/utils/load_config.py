import pathlib
import pathlib as path
import yaml

PROJECT_PATH = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_PATH / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["paths"]["root"] = str(PROJECT_PATH) + "/"
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)
