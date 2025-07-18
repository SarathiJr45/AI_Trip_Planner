import yaml
import os

def load_config(config_path:str ="config/config.yaml" )->dict:
    with open(config_path,"r") as f:
        config=yaml.safe_load(f)
    return config