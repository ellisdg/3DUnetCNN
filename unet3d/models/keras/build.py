from .resnet.resnet import ResnetBuilder
from .senet.se_resnet import SEResnetBuilder


def build_model(model_name, *args, **kwargs):
    if "encoder" in model_name.lower():
        if model_name[:2] == "se":
            builder = SEResnetBuilder
        else:
            builder = ResnetBuilder
        return getattr(builder, 'build_' + model_name.lower())(*args, **kwargs)
    else:
        raise ValueError(model_name + " is not a valid model name")
