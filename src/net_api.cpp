#include "net_api.hpp"

#include <boost/beast/core/detail/base64.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

namespace miner::net_api {

   std::string last_coin;

   std::uint8_t fetch_difficulty(void) {
      return 9; // TODO
   }


   std::string fetch_last_coin(void) {
      return last_coin;
   }


   void claim_coin(const FoundCoin& found) {
      std::cout << '\n';
      std::cout << "\nid_of_miner: " << found.id_of_miner;
      std::cout << "\ncoin_blob: "; print_hex_bytes(std::cout, found.coin_blob);
      std::cout << "\ndigest: "; print_hex_bytes(std::cout, found.digest);
      std::stringstream coin_blob_ss;
      print_hex_bytes(coin_blob_ss, found.coin_blob);
      last_coin = coin_blob_ss.str();
      {
         std::ofstream file(PATH.wallet_dir + PATH.coins_file, std::fstream::app);
         file << '\n' << last_coin;
      } {
         std::ofstream file(PATH.wallet_dir + PATH.last_coin_found_hash_file, std::fstream::trunc);
         print_hex_bytes(file, found.digest);
      }

      namespace base64 = boost::beast::detail::base64;
      std::string coin_blob_base64;
      coin_blob_base64.resize(base64::encoded_size(found.coin_blob.size()));
      base64::encode(coin_blob_base64.data(), found.coin_blob.data(), found.coin_blob.size());
      std::cout << "\ncoin_blob base64: " << coin_blob_base64;
      std::cout << '\a' << std::flush;

      // TODO yeet this string over through the API
   }
}