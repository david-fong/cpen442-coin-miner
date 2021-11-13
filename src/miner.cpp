#include "miner.hpp"
#include "net_api.hpp"

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <optional>

namespace miner {

   void print_digest(std::ostream& os, const std::span<std::uint8_t> digest) {
      const auto old_fmt = os.flags();
      os << std::hex << std::setfill('0');
      os << '\n';
      for (unsigned i = 0; i < digest.size(); i++) {
         os << std::setw(2) << static_cast<unsigned>(digest[i]);
      }
      os << std::flush;
      os.setf(old_fmt);
   }

   struct MinerThreadParams {
      std::uint8_t difficulty;
      const SHA256_CTX& hasher_prefix;
      std::string id_of_miner; // oof. wish this wasn't a hex string.
      std::uint8_t thread_num;
      std::uint8_t num_threads;
   };

   struct MinerThreadsResult {
      std::string id_of_miner;
      CoinBlob coin_blob;
      Digest digest;
   };

   struct ThreadFunc final {
      const MinerThreadParams params;
      std::mutex& result_mutex;
      bool& stop;
      std::optional<MinerThreadsResult>& result;

      void operator()();
      CoinBlob coin_blob_ = {0};
      static bool check_success(std::uint8_t difficulty, std::span<std::uint8_t>);
      void permute_coin_blob();
   };


   void ThreadFunc::operator()() {
      coin_blob_.back() = (0x100u * static_cast<unsigned>(params.thread_num)) / params.num_threads;
      coin_blob_[3] = 0x31u;
      while (!stop) {
         SHA256_CTX hasher = params.hasher_prefix;
         SHA256_Update(&hasher, coin_blob_.data(), coin_blob_.size());
         SHA256_Update(&hasher, params.id_of_miner.data(), params.id_of_miner.size());
         Digest md;
         SHA256_Final(md.data(), &hasher);
         if (check_success(params.difficulty, md)) [[unlikely]] {
            result_mutex.lock();
            result = MinerThreadsResult {
               .id_of_miner {params.id_of_miner},
               .coin_blob {coin_blob_},
               .digest {md}
            };
            stop = true;
            result_mutex.unlock();
            return;
         }
         permute_coin_blob();

         if (check_success(6, coin_blob_)) [[unlikely]] {
            // periodically print the coin_blob for debugging purposes.
            result_mutex.lock();
            print_digest(std::cout, coin_blob_);
            result_mutex.unlock();
         }
      }
   }


   bool ThreadFunc::check_success(const std::uint8_t difficulty, const std::span<std::uint8_t> data) {
      for (unsigned i = 0; i < difficulty/2; i++) {
         if (data[i]) [[likely]] { return false; }
      }
      if ((difficulty%2) && (data[difficulty/2] & 0xf0)) [[likely]] { return false; }
      return true;
   }


   void ThreadFunc::permute_coin_blob() {
      for (unsigned i = 0; i < COIN_BLOB_BYTES; i++) {
         if (i == COIN_BLOB_BYTES-1) [[unlikely]] {
            // TODO handle endpoint for this thread based on thread_num and num_threads
            ;
         }
         if (coin_blob_[i] == 0xff) [[unlikely]] {
            coin_blob_[i] = 0;
         } else {
            coin_blob_[i]++;
            break;
         }
      }
   }


   void MinerParams::clean() {
      num_threads = std::min(num_threads, static_cast<std::uint8_t>(std::thread::hardware_concurrency()));
   }


   void mine_coins(const MinerParams params) {
      // unsigned fetch_challenge_period_counter = 0;
      std::uint8_t difficulty = net_api::fetch_difficulty();
      std::string last_coin = net_api::fetch_last_coin();
      std::cout << "\ndifficulty: " << static_cast<unsigned>(difficulty);
      std::cout << "\nlast_coin: " << last_coin;
      std::cout.flush();
      /* while (true) */{
         SHA256_CTX hasher_prefix;
         SHA256_Init(&hasher_prefix);
         SHA256_Update(&hasher_prefix, CHALLENGE_PREFIX.data(), CHALLENGE_PREFIX.size());
         SHA256_Update(&hasher_prefix, last_coin.data(), last_coin.size());

         bool threads_stop = false;
         std::optional<MinerThreadsResult> threads_result = std::nullopt;
         std::mutex threads_result_mutex;
         std::vector<std::thread> threads;
         for (std::uint8_t i = 0; i < params.num_threads; i++) {
            threads.push_back(std::thread(ThreadFunc{
               .params {
                  .difficulty {difficulty},
                  .hasher_prefix {hasher_prefix},
                  .id_of_miner {params.id_of_miner},
                  .thread_num {i},
                  .num_threads {params.num_threads}
               },
               .result_mutex {threads_result_mutex},
               .stop {threads_stop},
               .result {threads_result},
            }));
         }
         /* while (!threads_result.stop) {
            std::this_thread::sleep_for(params.challenge_fetch_period);
            if (threads_result.stop) { break; }
            const std::string new_last_coin = net_api::fetch_last_coin();
            if (new_last_coin != last_coin) {
               threads_result.stop = true;
               last_coin = new_last_coin;
               difficulty = net_api::fetch_difficulty();
               break;
            }
         } */
         for (auto& thread : threads) {
            thread.join();
         }
         if (threads_result.has_value()) {
            auto result = threads_result.value();
            std::cout << "\nid_of_miner: " << result.id_of_miner;
            std::cout << "\ncoin_blob: "; print_digest(std::cout, result.coin_blob);
            std::cout << "\ndigest: "; print_digest(std::cout, result.digest);
            std::cout.flush();

            net_api::claim_coin(result.coin_blob);

            difficulty = net_api::fetch_difficulty();
            last_coin = net_api::fetch_last_coin();
         }
      }
   }
}