import os
import pickle
from typing import List, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from pandas import read_csv
# from databaseutils import get_hash, save_data_on_couchdb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from web3 import Web3
from web3.gas_strategies.rpc import rpc_gas_price_strategy

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

load_dotenv()


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    n_classes = 2
    n_features = 2
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_dataset() -> Dataset:
    raw_df = read_csv('./dataSet.csv')

    df = raw_df[['frequency',
                 'dose']]
    labels = raw_df[['target']]

    np_dataset = df.values

    X_train, X_test, y_train, y_test = train_test_split(
        np_dataset, labels, test_size=0.1, random_state=10)

    return (X_train, y_train), (X_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


full_abi = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "",
                "type": "string"
            }
        ],
        "name": "entries",
        "outputs": [
                {
                    "internalType": "string",
                    "name": "node",
                    "type": "string"
                }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "cid",
                "type": "string"
            }
        ],
        "name": "getByNode",
        "outputs": [
                {
                    "components": [
                        {
                            "internalType": "uint256",
                            "name": "round",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "metrics",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "model",
                            "type": "string"
                        }
                    ],
                    "internalType": "struct Flcontract.Round[]",
                    "name": "",
                    "type": "tuple[]"
                }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            },
            {
                "internalType": "string",
                "name": "metrics",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "model",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "cid",
                "type": "string"
            }
        ],
        "name": "send",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
full_address = '0x59E4fD714b73B733cD8d1c66f82238e087257C29'

min_abi = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "",
                                "type": "string"
            }
        ],
        "name": "entries",
        "outputs": [
            {
                "internalType": "string",
                "name": "node",
                "type": "string"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "cid",
                                "type": "string"
            }
        ],
        "name": "getByNode",
        "outputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "round",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "datahash",
                        "type": "string"
                    }
                ],
                "internalType": "struct Flcontract.Round[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "round",
                                "type": "uint256"
            },
            {
                "internalType": "string",
                "name": "datahash",
                                "type": "string"
            },
            {
                "internalType": "string",
                "name": "cid",
                                "type": "string"
            }
        ],
        "name": "send",
        "outputs": [],
        "stateMutability": "nonpayable",
                "type": "function"
    }
]
min_address = '0x850F95B0f32E9dB5AA484d160CB58f8A52103dc2'


def send_data(round, metrics, model, cid):
    print("Sending data to the network")
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    w3.eth.set_gas_price_strategy(rpc_gas_price_strategy)
    contract = w3.eth.contract(address=full_address, abi=full_abi)

    nonce = w3.eth.getTransactionCount(
        "0xf17f52151EbEF6C7334FAD080c5704D77216b732")

    # hash = get_hash(cid, round, model)

    transaction = contract.functions.send(round, metrics, model, cid).buildTransaction({
        "chainId": 1337,
        "from": "0xf17f52151EbEF6C7334FAD080c5704D77216b732",
        "gasPrice": w3.eth.gas_price,
        "nonce": nonce
    })

    signed_txn = w3.eth.account.sign_transaction(
        transaction, private_key=os.getenv('PRIVATE_ENV'))

    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("RECEIPT:", receipt)
    
    # save_data_on_couchdb(hash, cid, round, metrics, model)


def send_data_array(data_array):
    for data in data_array:
        send_data(data["round"], str(pickle.dumps(
            data["metrics"])), str(data["model"]), data["cid"])


def send_server_data_async(round, metrics, model):
    send_data(round, metrics, model, "server")
