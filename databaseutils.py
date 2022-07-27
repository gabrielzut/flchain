import pickle
from hashlib import sha256

import couchdb

couch = couchdb.Server()
couch = couchdb.Server('http://admin:admin@0.0.0.0:5984/')
db = couch['flchain']


def save_data_on_couchdb(hash, cid, round, metrics, model):
    db.save({
        "_id": hash,
        "cid": cid,
        "round": round,
        "metrics": metrics,
        "model": model
    })


def get_hash(cid, round, model):
    return sha256(pickle.dumps({"cid": cid, "round": round, "model": model})).hexdigest()
