#include "miner.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>

namespace miner {

	namespace difficulty {
		unsigned drop_cache_difficulty = MIN_DIFFICULTY - 1;
		void init_drop_cache(void) {
			if (std::filesystem::exists(PATH.wallet_dir + PATH.difficulty_drop_cache_file)) {
				std::ifstream file(PATH.wallet_dir + PATH.difficulty_drop_cache_file);
				file >> drop_cache_difficulty;
			}
		}
	}

	void print_hex_bytes(std::ostream& os, const std::span<const std::uint8_t> bytes) {
		const auto old_fmt = os.flags();
		os << std::hex << std::setfill('0');
		for (unsigned i = 0; i < bytes.size(); i++) {
			os << std::setw(2) << static_cast<unsigned>(bytes[i]);
		}
		os << std::flush;
		os.setf(old_fmt);
	}


	struct MinerThreadParams final {
		unsigned difficulty;
		const SHA256_CTX& hasher_prefix;
		std::string id_of_miner; // oof. wish this wasn't a hex string.
		std::string team_member_id;
		std::uint8_t seed;
		unsigned thread_num;
		unsigned num_threads;
	};


	struct MinerThreadsSharedData final {
		mutable std::mutex mutex;
		bool stop = false;
		FoundCoin found_coin;
	};


	struct ThreadFunc final {
		const MinerThreadParams params;
		MinerThreadsSharedData& share;

		void operator()();
		CoinBlob coin_blob_ = {0};
		static unsigned count_leading_zero_nibbles_(unsigned difficulty, std::span<std::uint8_t>);
		void permute_coin_blob_();
	};


	void ThreadFunc::operator()() {
		for (unsigned i = 0; i < std::min(TEAM_MEMBER_ID_BYTES, params.team_member_id.size()); i++) {
			coin_blob_[COIN_BLOB_TEAM_MEMBER_ID_OFFSET+i] = params.team_member_id[i];
		}
		coin_blob_[COIN_BLOB_TEAM_MEMBER_ID_OFFSET-1] = params.seed;
		coin_blob_[COIN_BLOB_TEAM_MEMBER_ID_OFFSET-2] = static_cast<std::uint8_t>(
			(0x100u * params.thread_num) / params.num_threads
		);
		while (!share.stop) {
			SHA256_CTX hasher = params.hasher_prefix;
			SHA256_Update(&hasher, coin_blob_.data(), coin_blob_.size());
			SHA256_Update(&hasher, params.id_of_miner.data(), params.id_of_miner.size());
			Digest md;
			SHA256_Final(md.data(), &hasher);
			if (share.stop) { return; }
			const unsigned found_difficulty = count_leading_zero_nibbles_(params.difficulty, md);
			if (found_difficulty >= params.difficulty) [[unlikely]] {
				const std::lock_guard mutex_guard(share.mutex);
				share.found_coin = FoundCoin {
					.id_of_miner {params.id_of_miner},
					.coin_blob {coin_blob_},
					.digest {md}
				};
				share.stop = true;
				return;

			} else if (found_difficulty > difficulty::drop_cache_difficulty) [[unlikely]] {
				const std::lock_guard mutex_guard(share.mutex);
				difficulty::drop_cache_difficulty = found_difficulty;
				std::ofstream file(PATH.wallet_dir + PATH.difficulty_drop_cache_file, std::fstream::trunc);
				file << found_difficulty << ' ';
				print_hex_bytes(file, coin_blob_);
			}
			permute_coin_blob_();

			// periodically print the coin_blob for debugging purposes:
			/* if (count_leading_zero_nibbles_(6, coin_blob_) >= 6) [[unlikely]] {
				const std::lock_guard mutex_guard(share.mutex);
				if (share.stop) { return; }
				std::clog << "\nthread " << params.thread_num << " progress: ";
				print_hex_bytes(std::clog, coin_blob_);
			} */
		}
	}


	unsigned ThreadFunc::count_leading_zero_nibbles_(const unsigned difficulty, const std::span<std::uint8_t> data) {
		unsigned count = 0;
		unsigned i = 0;
		for (; i < difficulty/2; i++) {
			if (data[i]) [[likely]] {
				if (!(data[i] & 0xf0)) [[likely]] { count += 1; }
				return count;
			}
			else { count += 2; }
		}
		if (!(data[i] & 0xf0)) [[likely]] { count += 1; }
		return count;
	}


	void ThreadFunc::permute_coin_blob_() {
		for (unsigned i = 0; i < COIN_BLOB_TEAM_MEMBER_ID_OFFSET; i++) {
			if (i == COIN_BLOB_TEAM_MEMBER_ID_OFFSET-1) [[unlikely]] {
				// TODO handle endpoint for this thread based on thread_num and num_threads
				// well... it's almost astronomically unlikely that the permutation will
				// reach this point...
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
		team_member_id.resize(TEAM_MEMBER_ID_BYTES);
		num_threads = std::min(num_threads, std::thread::hardware_concurrency());
	}


	void mine_coin(const MinerParams params) {
		difficulty::init_drop_cache();
		{
			std::ostream& os = std::clog;
			// os << "\nid_of_miner: " << params.id_of_miner;
			os << "\nteam_member_id: " << params.team_member_id;
			os << "\nlast_coin: " << params.last_coin;
			os << "\ndifficulty: " << params.difficulty;
			os << std::endl;
		}
		// a naive seed for handling restarts since the coin_blob is a counter:
		std::uint8_t seed; {
			std::ifstream file(PATH.wallet_dir + PATH.seed_file);
			if (file.fail()) {
				seed = 0;
			} else {
				file >> seed;
			}
		} {
			std::ofstream file(PATH.wallet_dir + PATH.seed_file, std::fstream::trunc);
			file << static_cast<std::uint8_t>(seed + 1);
		}

		SHA256_CTX hasher_prefix;
		SHA256_Init(&hasher_prefix);
		SHA256_Update(&hasher_prefix, CHALLENGE_PREFIX.data(), CHALLENGE_PREFIX.size());
		SHA256_Update(&hasher_prefix, params.last_coin.data(), params.last_coin.size());

		MinerThreadsSharedData threads_shared_data;
		typename std::vector<std::thread> threads;
		for (std::uint8_t i = 0; i < params.num_threads; i++) {
			threads.push_back(std::thread(ThreadFunc{
				.params {
					.difficulty = params.difficulty,
					.hasher_prefix {hasher_prefix},
					.id_of_miner {params.id_of_miner},
					.team_member_id {params.team_member_id},
					.seed = seed,
					.thread_num = i,
					.num_threads = params.num_threads
				},
				.share {threads_shared_data},
			}));
		}
		for (auto& thread : threads) {
			thread.join();
		}
		claim_coin(threads_shared_data.found_coin);
	}


	void claim_coin(const FoundCoin& found) {
		{
			std::ostream& os = std::clog;
			os << '\n';
			os << "\nid_of_miner: " << found.id_of_miner;
			os << "\ncoin_blob: "; print_hex_bytes(os, found.coin_blob);
			os << "\ndigest: "; print_hex_bytes(os, found.digest);
		} {
			std::ofstream file(PATH.wallet_dir + PATH.coins_file, std::fstream::app);
			file << '\n';
			print_hex_bytes(file, found.coin_blob);
		} {
			std::ofstream file(PATH.wallet_dir + PATH.last_coin_found_hash_file, std::fstream::trunc);
			print_hex_bytes(file, found.digest);
		}
		print_hex_bytes(std::cout, found.coin_blob);
		std::clog << '\a' << std::flush;
	}
}