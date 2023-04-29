from src.main import bayesnet, preprocess

bnet = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)
bnet.build_network()


def test_inference():
    pass
