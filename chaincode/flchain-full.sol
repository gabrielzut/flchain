// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.7.0 <0.9.0;

contract Flcontract {
    struct Round {
        uint256 round;
        string metrics;
        string model;
    }

    struct Entry {
        string node;
        Round[] rounds;
    }

    address source;
    mapping(string => Entry) public entries;

    constructor() {
        source = msg.sender;
    }

    function send(
        uint256 round,
        string memory metrics,
        string memory model,
        string memory cid
    ) public {
        require(round >= 0, "round must be bigger than 0");
        Entry storage entry = entries[cid];

        if (bytes(entry.node).length == 0) {
            entry.node = cid;
        }

        //entry.rounds[round] = Round(round, accuracy, loss);
        entry.rounds.push(Round(round, metrics, model));
        entries[cid] = entry;
    }

    function getByNode(string memory cid) public view returns (Round[] memory) {
        return entries[cid].rounds;
    }
}
