from pomegranate import *
from importlib import reload
import json
import os

import preprocess

# from src.main import preprocess

reload(preprocess)
import pygraphviz
import pandas as pd


class BayesNet(BayesianNetwork):
    def __init__(
        self,
        structure: tuple = None,
        data: pd.DataFrame = None,
        psuedo_data: bool = False,
    ):
        self.struct = None
        self.data = None
        self.network = None
        self.code_to_idx = {}  # code to index mapping
        super(BayesianNetwork, self).__init__()
        if (not structure or not data) and psuedo_data:
            # i.e. there is no element(s) passed then we simply use the preprocess function we made to make the data
            self.struct, self.data = preprocess.make_data()
        else:
            # if they pass it through then thats good
            pass

    def build_network(self):
        assert self.struct is not None, "You have not defined structure or data yet"
        assert self.data is not None, "You have not defined structure or data yet"

        def build_idx_mapping():
            """This method builds the dictionary that is used to lookup the index of a specific
            element that needs to be accessed to check probabilities after performing inference
            """
            for idx, state in enumerate(self.network.states):
                self.code_to_idx[state.name] = idx

        self.network = BayesianNetwork.from_structure(
            self.data.to_numpy(),
            structure=self.struct,
            state_names=self.data.columns.values,
            name="model",
        )
        self.network.bake()
        build_idx_mapping()

    def plot_network(self):
        assert self.network is not None, "There's no internally defined network yet"
        self.network.plot()

    def infer_proba(self, preds: dict):
        """perform inference. Pass in a DICTIONARY simply containing the node (or nodes) from the model states
        and the probability of observing that specific node

        example:
        preds = {'C': 0.75}

        This says that we're passing in C being true as a value of 0.75 and performing inference on the children OF c
        """

        # need to convert the probability into a discrete distribution then pass that into model predictions
        def proba_to_distribution(proba):
            return DiscreteDistribution({1: proba, 0: 1 - proba})

        new_pred_vals = list(map(proba_to_distribution, list(preds.values())))
        keys = preds.keys()
        preds_passing_in = dict(zip(keys, new_pred_vals))
        preds = self.network.predict_proba(preds_passing_in)
        return preds

    def save_model(self):
        """Serializes model to json and saves it in a specified location"""
        j = self.network.to_json()
        with open("model.json", "w") as f:
            json.dump(j, f)

    def get_idx_mapping(self):
        return self.code_to_idx

    def load_model(self, dir, model_name):
        """Loads json model from the directory it's currently in"""
        load_dir = os.path.join(dir, model_name)
        f = open(load_dir, "r")
        # Reading from file
        data = json.loads(f.read())
        self.network = BayesianNetwork.from_json(data)
        self.network.bake()

    @classmethod
    def get_pred_proba(self, node, predictions, code_to_idx):
        """Pass in a given node and the predictions and it'll parse the object and return the
        probability that it's true. This returns a tuple of (true proba, false proba)"""
        preds = predictions[code_to_idx[node]]

        true_proba = preds.parameters[0][1]
        false_proba = 1 - true_proba

        return true_proba, false_proba
