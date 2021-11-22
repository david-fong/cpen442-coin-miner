#ifndef HPP_MINER
#define HPP_MINER

#include <openssl/sha.h>
#include <iosfwd>
#include <string>
#include <array>
#include <span>
#include <cstdint>

namespace miner {

	const struct {
		std::string wallet_dir = "wallet/";
		std::string coins_file = "coins.txt";
		std::string last_coin_found_hash_file = "last_coin_found_hash.txt";
		std::string seed_file = "seed.txt";
	} PATH;

	const std::string CHALLENGE_PREFIX = "CPEN 442 Coin2021";
	constexpr size_t TEAM_MEMBER_ID_BYTES = 7; // chosen to fit all our first names
	constexpr size_t COIN_BLOB_BYTES = 38;
	constexpr size_t COIN_BLOB_TEAM_MEMBER_ID_OFFSET = COIN_BLOB_BYTES-TEAM_MEMBER_ID_BYTES;
	using CoinBlob = std::array<std::uint8_t, COIN_BLOB_BYTES>;
	using Digest = std::array<std::uint8_t, SHA256_DIGEST_LENGTH>;

	struct MinerParams {
		std::string id_of_miner;
		std::string team_member_id;
		std::string last_coin;
		unsigned difficulty;
		unsigned num_threads;

		void clean();
	};

	struct FoundCoin {
		std::string id_of_miner;
		CoinBlob coin_blob;
		Digest digest;
	};

	void print_hex_bytes(std::ostream&, const typename std::span<const std::uint8_t>);

	void mine_coin(MinerParams);

	void claim_coin(const FoundCoin&);
}
#endif