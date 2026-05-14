from utils.read_settings import load_config
from utils.general import colorstr
from detection_script_v2 import run

# for khadas_camera_to_undistort it
class MyClass():
    def __init__(self, param):
        self.param = param


def read_settings():
    settings = load_config(file_path="config.yaml", section="settings")
    settings["imgsz"] *= 2 if len(settings["imgsz"]) == 1 else 1  # expand
    return dict(settings)


def main():
    settings = read_settings()
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in settings.items()))
    run(**settings)


if __name__ == "__main__":
    main()