#ifndef HPP_MINER
#define HPP_MINER

#include <iosfwd>
#include <string>
#include <array>
#include <chrono>
#include <span>
#include <cstdint>
#include <openssl/sha.h>

namespace miner {
   const std::string CHALLENGE_PREFIX = "CPEN 442 Coin2021";
   constexpr size_t COIN_BLOB_BYTES = 38;
   using CoinBlob = std::array<std::uint8_t, COIN_BLOB_BYTES>;
   using Digest = std::array<std::uint8_t, SHA256_DIGEST_LENGTH>;

   struct MinerParams {
      // const unsigned fetch_challenge_period = 
      std::chrono::seconds challenge_fetch_period;
      std::uint8_t num_threads;
      std::string id_of_miner;

      void clean();
   };

   void print_digest(std::ostream&, const typename std::span<std::uint8_t>);

   void mine_coins(MinerParams);
}
#endif