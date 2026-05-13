import yaml

def load_config(file_path="config.yaml", section="settings"):
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            # Return just the requested section as a dictionary
            return config.get(section, {})
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}