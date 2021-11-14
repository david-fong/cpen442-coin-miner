#include "miner.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * argv[1] id_of_miner
 * argv[2] team_member_id (7 bytes)
 * argv[3] last_coin
 * argv[4] difficulty
 * argv[5] num_threads
 */
int main(const int argc, char const *const argv[]) {
	// My implementation specifies this as safe:
	std::ios_base::sync_with_stdio(false);
	if (argc < 4) {
		std::cerr << "missing args. check main.cpp for args format.";
		return 1;
	}

	miner::MinerParams miner_params {
		.id_of_miner = std::string(argv[1]),
		.team_member_id = std::string(argv[2]),
		.last_coin = std::string(argv[3]),
		.difficulty = static_cast<unsigned>(std::stoi(argv[4])),
		.num_threads = static_cast<unsigned>(std::stoi(argv[5])),
	};
	miner_params.clean();
	miner::mine_coin(miner_params);
	return 0;
}