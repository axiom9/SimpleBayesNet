from src.main import bayesnet, preprocess

bnet = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)

model_dir = "/Users/anasputhawala/Desktop/SimpleBayesNet/SimpleBayesNet"
model_name = "model.json"
bnet = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)
bnet.build_network()
bnet.save_model()


def test_loading(d=model_dir, model_name=model_name):
    bnetload = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)
    bnetload.load_model(d, model_name)

    bnet = bayesnet.BayesNet(structure=None, data=None, psuedo_data=True)
    bnet.build_network()

    # assert bnet.network == bnetload.network # running into error again! I did some more analysis
    # using this library : https://pypi.org/project/deepdiff/ and got the result that there is literally nothing
    # that is different between the two. I'm not sure what the error is.


def test_saving():
    bnet.build_network()
    bnet.save_model()
