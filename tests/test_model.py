from H2L.trainer import cnns
from H2L.trainer import resnet
from keras.models import Model


def test_build_cnn_sequence():
    model, paras = cnns.sequentialModel(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)


def test_build_cnn_branch():
    model, paras = cnns.sequentialModel(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)


def test_build_res():
    model, paras = resnet.res32(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)
