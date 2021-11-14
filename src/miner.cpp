#include "miner.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <algorithm>

namespace miner {

	void print_hex_bytes(std::ostream& os, const std::span<const std::uint8_t> digest) {
		const auto old_fmt = os.flags();
		os << std::hex << std::setfill('0');
		for (unsigned i = 0; i < digest.size(); i++) {
			os << std::setw(2) << static_cast<unsigned>(digest[i]);
		}
		os << std::flush;
		os.setf(old_fmt);
	}


	struct MinerThreadParams final {
		unsigned difficulty;
		const SHA256_CTX& hasher_prefix;
		std::string id_of_miner; // oof. wish this wasn't a hex string.
		std::string team_member_id;
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
		static bool check_success_(std::uint8_t difficulty, std::span<std::uint8_t>);
		void permute_coin_blob_();
	};


	void ThreadFunc::operator()() {
		for (unsigned i = 0; i < std::min(TEAM_MEMBER_ID_BYTES, params.team_member_id.size()); i++) {
			coin_blob_[COIN_BLOB_TEAM_MEMBER_ID_OFFSET+i] = params.team_member_id[i];
		}
		coin_blob_[COIN_BLOB_TEAM_MEMBER_ID_OFFSET-1] = (0x100u * params.thread_num) / params.num_threads;
		while (!share.stop) {
			SHA256_CTX hasher = params.hasher_prefix;
			SHA256_Update(&hasher, coin_blob_.data(), coin_blob_.size());
			SHA256_Update(&hasher, params.id_of_miner.data(), params.id_of_miner.size());
			Digest md;
			SHA256_Final(md.data(), &hasher);
			if (share.stop) { return; }
			if (check_success_(params.difficulty, md)) [[unlikely]] {
				const std::lock_guard mutex_guard(share.mutex);
				share.found_coin = FoundCoin {
					.id_of_miner {params.id_of_miner},
					.coin_blob {coin_blob_},
					.digest {md}
				};
				share.stop = true;
				return;
			}
			permute_coin_blob_();

			// periodically print the coin_blob for debugging purposes:
			if (check_success_(6, coin_blob_)) [[unlikely]] {
				const std::lock_guard mutex_guard(share.mutex);
				if (share.stop) { return; }
				std::clog << "\nthread " << params.thread_num << " progress: ";
				print_hex_bytes(std::clog, coin_blob_);
			}
		}
	}


	bool ThreadFunc::check_success_(const std::uint8_t difficulty, const std::span<std::uint8_t> data) {
		for (unsigned i = 0; i < difficulty/2; i++) {
			if (data[i]) [[likely]] { return false; }
		}
		if ((difficulty%2) && (data[difficulty/2] & 0xf0)) [[likely]] { return false; }
		return true;
	}


	void ThreadFunc::permute_coin_blob_() {
		for (unsigned i = 0; i < COIN_BLOB_TEAM_MEMBER_ID_OFFSET; i++) {
			if (i == COIN_BLOB_TEAM_MEMBER_ID_OFFSET) [[unlikely]] {
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
		{
			std::ostream& os = std::clog;
			os << "\nid_of_miner: " << params.id_of_miner;
			os << "\nteam_member_id: " << params.team_member_id;
			os << "\nlast_coin: " << params.last_coin;
			os << "\ndifficulty: " << params.difficulty;
			os.flush();
		}
	
		SHA256_CTX hasher_prefix;
		SHA256_Init(&hasher_prefix);
		SHA256_Update(&hasher_prefix, CHALLENGE_PREFIX.data(), CHALLENGE_PREFIX.size());
		SHA256_Update(&hasher_prefix, params.last_coin.data(), params.last_coin.size());

		MinerThreadsSharedData threads_shared_data;
		std::vector<std::thread> threads;
		for (std::uint8_t i = 0; i < params.num_threads; i++) {
			threads.push_back(std::thread(ThreadFunc{
				.params {
					.difficulty {params.difficulty},
					.hasher_prefix {hasher_prefix},
					.id_of_miner {params.id_of_miner},
					.team_member_id {params.team_member_id},
					.thread_num {i},
					.num_threads {params.num_threads}
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
		std::cout << '\n';
		std::cout << "\nid_of_miner: " << found.id_of_miner;
		std::cout << "\ncoin_blob: "; print_hex_bytes(std::cout, found.coin_blob);
		std::cout << "\ndigest: "; print_hex_bytes(std::cout, found.digest);
		std::stringstream coin_blob_ss;
		print_hex_bytes(coin_blob_ss, found.coin_blob);
		const std::string coin_blob_str = coin_blob_ss.str();
		{
			std::ofstream file(PATH.wallet_dir + PATH.coins_file, std::fstream::app);
			file << '\n' << coin_blob_str;
		} {
			std::ofstream file(PATH.wallet_dir + PATH.last_coin_found_hash_file, std::fstream::trunc);
			print_hex_bytes(file, found.digest);
		}
		std::cout << '\a' << std::flush;
	}
}