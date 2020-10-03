

def run_training_with_package(package="keras", *args, **kwargs):
    """
    Agnostic function to run the training with the specified deep learning package/framework.
    :param package:
    :param args:
    :param kwargs:
    :return:
    """
    if package == "keras":
        from .keras import run_keras_training as run_training
    elif package == "pytorch":
        from .pytorch import run_pytorch_training as run_training
    else:
        raise RuntimeError("{} package is not supported".format(package))
    return run_training(*args, **kwargs)
