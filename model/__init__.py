from .lightning_yolov3 import YoloLight

__all__ = {
    "YOLO": YoloLight

}

def create_model(config):
    return __all__[config["name"]](config)