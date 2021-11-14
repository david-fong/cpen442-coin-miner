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
      unsigned num_threads;
      std::string id_of_miner;

      void clean();
   };

   struct FoundCoin {
      std::string id_of_miner;
      CoinBlob coin_blob;
      Digest digest;
   };

   void print_hex_bytes(std::ostream&, const typename std::span<const std::uint8_t>);

   void mine_coins(MinerParams);
}
#endif