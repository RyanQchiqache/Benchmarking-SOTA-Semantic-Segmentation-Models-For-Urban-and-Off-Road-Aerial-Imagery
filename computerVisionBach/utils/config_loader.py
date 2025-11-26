import os
from omegaconf import OmegaConf

def load_config(name="config.yaml"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "configs", name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg
