import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import parameters_to_weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from flwr.common.typing import Parameters, Scalar
from flwr.server.server import FitResultsAndFailures

import utils


class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_weights, metrics_aggregated = super().aggregate_fit(rnd,
                                                                       results, failures)

        if aggregated_weights is not None:
            client_params = []
            for tuple in results:
                client_params.append(
                    {"cid": tuple[0].cid, "metrics": {"accuracy": tuple[1].metrics["accuracy"], "loss": tuple[1].metrics["loss"], "round": rnd}, "model": tuple[1].metrics["model"], "round": rnd})

            utils.send_data_array(client_params)

        model = LogisticRegression()
        utils.set_model_params(
            model, parameters_to_weights(aggregated_weights))

        model_save = pickle.dumps(model)

        utils.send_server_data_async(rnd, "", model_save)

        return aggregated_weights, metrics_aggregated


def on_fit_fn(rnd: int) -> Dict:
    print("Sending round number:", rnd)
    print("Round start: ", int(round(time.time() * 1000)))
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    _, (X_test, y_test) = utils.load_dataset()

    def evaluate(parameters: fl.common.Weights):
        print(parameters)

        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)

        print("Round after evaluate: ", int(round(time.time() * 1000)))

        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = CustomStrategy(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=on_fit_fn,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config={"num_rounds": 5},
    )
