# flchain

Federated learning + blockchain example


**HOW TO CONFIGURE**
- Set up a network with quorum-test-network or similar;
- Place a .env file with the PRIVATE_ENV (private key of the node that sends the transactions) in this folder;
- Deploy the contract (the full version sends the models to the ledger and the min version only the hash);
- Change the send_data, send_data_array and the send_server_data function (utils.py) so it reflects the functions of the contract you chose;
- If you chose the min contract, uncomment the line with save_data_on_couchdb (send_data function);
- Install the dependencies with Poetry (poetry shell && poetry install)
- Run a server with poetry run python3 server.py and some clients with poetry run python3 client.py

If you run into the "undefined symbol: _PyGen_Send" error, change the Python version to 3.9 (like "poetry env use /usr/bin/python3.9")