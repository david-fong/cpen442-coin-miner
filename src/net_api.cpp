#include "net_api.hpp"

namespace miner::net_api {

   std::uint8_t fetch_difficulty(void) {
      return 8; // TODO
   }


   std::string fetch_last_coin(void) {
      return "a9c1ae3f4fc29d0be9113a42090a5ef9fdef93f5ec4777a008873972e60bb532"; // TODO
   }


   void claim_coin(const std::span<std::uint8_t> coin_blob) {
      (void) coin_blob; // TODO
   }
}