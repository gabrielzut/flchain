import warnings
import flwr as fl
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = utils.load_dataset()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,
        warm_start=True,
    )

    utils.set_initial_params(model)

    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)

            results = {}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

                loss = log_loss(y_test, model.predict_proba(X_test))
                accuracy = model.score(X_test, y_test)

                # model_save = pickle.dumps(model)

                results = {
                    "loss": loss,
                    "accuracy": accuracy
                    # "model": model_save
                }

            print(f"Training finished for round {config['rnd']}")
            params = utils.get_model_parameters(model)

            return params, len(X_train), results

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("127.0.0.1:8080", client=MnistClient())
