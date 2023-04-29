from src.main import bayesnet, preprocess

bnet = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)


def test_type_struct():
    assert type(bnet.struct) == tuple


def test_struct_vals():
    struct, data = preprocess.make_data()
    assert bnet.struct == struct
    assert bnet.data.equals(data)
