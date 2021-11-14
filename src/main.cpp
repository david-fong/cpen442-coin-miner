#include "miner.hpp"
#include "net_api.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * argv[1] id_of_miner
 * argv[2] team_member_id
 * argv[3] num_threads
 */
int main(const int argc, char const *const argv[]) {
	// My implementation specifies this as safe:
	std::ios_base::sync_with_stdio(false);
   {
      using namespace miner::net_api;
      std::ifstream wallet(PATH.wallet_dir + PATH.last_coin_found_hash_file);
      wallet >> miner::net_api::last_coin;
   }
   if (argc < 4) {
      std::cerr << "missing args. check main.cpp for args format.";
      return 1;
   }

   miner::MinerParams miner_params {
      .challenge_fetch_period {std::chrono::seconds(10)},
      .num_threads {5},
      .id_of_miner = std::string(argv[1]),
   };
   miner_params.clean();
   miner::mine_coins(miner_params);

   (void)argc;
   (void)argv;
}