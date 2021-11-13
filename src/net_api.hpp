#ifndef HPP_NET_API
#define HPP_NET_API

#include "miner.hpp"

#include <string>
#include <vector>
#include <span>
#include <cstdint>

namespace miner::net_api {
   std::uint8_t fetch_difficulty(void);
   std::string fetch_last_coin(void);
   void claim_coin(std::span<std::uint8_t>);
}

#endif