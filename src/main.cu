/** Adapted from https://github.com/moffa13/SHA256CUDA */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sha256.cuh"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>


#define SHOW_INTERVAL_MS 10000
#define BLOCK_SIZE 256
#define SHA_PER_ITERATIONS 8'388'608
#define NUMBLOCKS (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE


static size_t difficulty = 1;

// Output string by the device read by host
uint8_t* g_nonce_out = nullptr;
uint8_t* g_hash_out = nullptr;
int* g_found = nullptr;

static uint64_t nonce = 0;
static uint64_t user_nonce = 0;
static uint64_t last_nonce_since_update = 0;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;


void print_hex_bytes(std::ostream& os, const uint8_t* bytes, size_t bytes_size) {
	os << std::hex << std::setfill('0');
	for (uint8_t i = 0; i < bytes_size; ++i) {
		os << std::setw(2) << static_cast<int>(bytes[i]);
	}
	os << std::dec << std::endl;
}


__device__ uint8_t count_leading_zero_nibbles_(const uint8_t* const data, const uint8_t difficulty) {
	unsigned count = 0;
	unsigned i = 0;
	for (; i < difficulty/2; i++) {
		if (data[i]) [[likely]] {
			if (!(data[i] & 0xf0)) { count += 1; }
			return count;
		}
		else { count += 2; }
	}
	if (!(data[i] & 0xf0)) { count += 1; }
	return count;
}


__device__ uint8_t nonce_to_bytes(uint64_t nonce, unsigned uint8_t* out) {
	for (unsigned i = 0; i < 8; i++) {
		out[i] = static_cast<uint8_t>(nonce >> (8 * i));
	}
	return 8;
}


extern __shared__ uint8_t threads_buffer[];
__global__ void sha256_kernel(
	uint8_t* out_nonce, uint8_t* out_found_hash,
	int *out_found,
	const SHA256_CTX* const hasher_prefix,
	const uint8_t difficulty, const uint64_t nonce_seed,
	const uint8_t* miner_id_in, const size_t miner_id_in_size
) {
	// If this is the first thread of the block, init the input string in shared memory
	uint8_t* const miner_id_in = &threads_buffer[0];
	if (threadIdx.x == 0) {
		memcpy(miner_id_in, miner_id_in, miner_id_in_size + 1);
	}
	__syncthreads(); // Ensure the strings have been written to SMEM

	// The first byte we can write because there is the miner_id string at the begining
	// Respects the memory padding of 8 bit (uint8_t).
	const size_t min_threads_buffer = static_cast<size_t>(std::ceil((miner_id_in_size + 1) / 8.f) * 8);

	const uintptr_t md_addr = threadIdx.x * (64) + min_threads_buffer;
	const uintptr_t nonce_addr = md_addr + 32;

	uint8_t* const md = &threads_buffer[md_addr];
	uint8_t* const nonce = &threads_buffer[nonce_addr];
	memset(nonce, 0, 32);

	nonce_to_bytes(nonce_seed + (blockIdx.x * blockDim.x + threadIdx.x), nonce);
	{
		SHA256_CTX ctx = *hasher_prefix;
		sha256_update(&ctx, nonce, 32);
		sha256_update(&ctx, miner_id_in, miner_id_in_size);
		sha256_final(&ctx, md);
	}
	if ((count_leading_zero_nibbles_(md, difficulty) >= difficulty) && (atomicExch(out_found, 1) == 0)) {
		memcpy(out_found_hash, md, 32);
		memcpy(out_nonce, nonce, 32);
	}
}


void print_state() {
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;

	if (last_show_interval.count() > SHOW_INTERVAL_MS) {
		std::chrono::duration<double, std::milli> span = t2 - t_last_updated;
		float ratio = span.count() / 1000;
		std::clog << span.count() << " " << nonce - last_nonce_since_update << std::endl;
		std::clog << std::fixed << static_cast<uint64_t>((nonce - last_nonce_since_update) / ratio) << " hashes/s" << std::endl;
		std::clog << std::fixed << "nonce: " << nonce << std::endl;

		t_last_updated = std::chrono::high_resolution_clock::now();
		last_nonce_since_update = nonce;
	}
}


int main(const int argc, char const *const argv[]) {
	cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	t_last_updated = std::chrono::high_resolution_clock::now();

	const std::string arg_id_of_miner(argv[1]);
	const std::string team_member_id(argv[2]); team_member_id.resize(8, '\0');
	const std::string last_coin(argv[3]);
	difficulty = std::stoi(argv[4]);
	// num_threads (ignored)

	const std::string prefix = std::string("CPEN 442 Coin2021") + last_coin;
	SHA256_CTX hasher_prefix;
	sha256_init(&hasher_prefix);
	sha256_update(&hasher_prefix, prefix.c_str(), prefix.size());
	sha256_update(&hasher_prefix, team_member_id.data(), team_member_id.size());

	std::clog << "Nonce : ";
	std::cin >> user_nonce;

	char* g_miner_id_str = nullptr;
	cudaMalloc(&g_miner_id_str, arg_id_of_miner.size()+1);
	cudaMemcpy(g_miner_id_str, arg_id_of_miner.c_str(), arg_id_of_miner.size()+1, cudaMemcpyHostToDevice);

	cudaMallocManaged(&g_nonce_out, 32);
	cudaMallocManaged(&g_hash_out, 32);
	cudaMallocManaged(&g_found, sizeof(int));
	*g_found = 0;

	nonce += user_nonce;
	last_nonce_since_update += user_nonce;

	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));

	const size_t dynamic_shared_size = (ceil((prefix.size() + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);
	std::clog << "Shared memory is " << dynamic_shared_size / 1024 << "KB" << std::endl;

	while (!*g_found) {
		sha256_kernel << < NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size >> > (
			g_nonce_out, g_hash_out, g_found, &hasher_prefix, difficulty, nonce, g_miner_id_str, arg_id_of_miner.size()
		);
		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error("Device error");
		}
		nonce += NUMBLOCKS * BLOCK_SIZE;
		print_state();
	}
	print_hex_bytes(std::clog, g_hash_out, 32);
	print_hex_bytes(std::cout, team_member_id.data(), team_member_id.size());
	print_hex_bytes(std::cout, g_nonce_out, 32);

	cudaFree(g_nonce_out);
	cudaFree(g_hash_out);
	cudaFree(g_found);
	cudaFree(g_miner_id_str);
	cudaDeviceReset();
	return 0;
}