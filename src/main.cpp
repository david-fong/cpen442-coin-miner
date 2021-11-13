#include "miner.hpp"
#include <string>
#include <iostream>
#include <chrono>
// 1469b33100000000000000000000000000000000000000000000000000000000000000000000
/*
{
  "coin_id": "a9c1ae3f4fc29d0be9113a42090a5ef9fdef93f5ec4777a008873972e60bb532",
  "id_of_miner": "genesis",
  "time_stamp": 1636620047
}
*/

int main(const int argc, char const *const argv[]) {
	// My implementation specifies this as safe:
	std::ios_base::sync_with_stdio(false);

   miner::MinerParams miner_params {
      .challenge_fetch_period {std::chrono::seconds(10)},
      .num_threads {5},
      .id_of_miner = "fbb80563372e11643f1b1cc89f47d3b45a36c20be12b9a23b37332b381c6c64",
   };
   miner_params.clean();
   miner::mine_coins(miner_params);

   (void)argc;
   (void)argv;
}