#ifndef HPP_NET_API
#define HPP_NET_API

#include "miner.hpp"

#include <string>
#include <vector>
#include <span>
#include <cstdint>

namespace miner::net_api {
   const struct {
      std::string wallet_dir = "wallet/";
      std::string coins_file = "coins.txt";
      std::string last_coin_found_hash_file = "last_coin_found_hash.txt";
   } PATH;
   extern std::string last_coin;
   std::uint8_t fetch_difficulty(void);
   std::string fetch_last_coin(void);
   void claim_coin(const FoundCoin&);
}
#endif