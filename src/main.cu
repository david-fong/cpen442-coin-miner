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
uint8_t* g_out = nullptr;
uint8_t* g_hash_out = nullptr;
int* g_found = nullptr;

static uint64_t nonce = 0;
static uint64_t user_nonce = 0;
static uint64_t last_nonce_since_update = 0;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;


void print_hash(const uint8_t* sha256) {
	std::cout << std::hex << std::setfill('0');
	for (uint8_t i = 0; i < 32; ++i) {
		std::cout << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
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


__device__ uint8_t nonce_to_str(uint64_t nonce, unsigned char* out) {
	uint64_t result = nonce;
	uint8_t remainder;
	uint8_t nonce_size = nonce == 0 ? 1 : floor(log10((double)nonce)) + 1;
	uint8_t i = nonce_size;
	while (result >= 10) {
		remainder = result % 10;
		result /= 10;
		out[--i] = remainder + '0';
	}

	out[0] = result + '0';
	i = nonce_size;
	out[i] = 0;
	return i;
}


extern __shared__ uint8_t threads_buffer[];
__global__ void sha256_kernel(
	uint8_t* out_input_string_nonce, uint8_t* out_found_hash,
	int *out_found,
	const uint8_t* prefix_in, size_t prefix_in_size,
	uint8_t difficulty, uint64_t nonce_seed,
	const * miner_id_in, const size_t miner_id_in_size
) {
	// If this is the first thread of the block, init the input string in shared memory
	uint8_t* const prefix = &threads_buffer[0];
	uint8_t* const miner_id_in = &threads_buffer[static_cast<size_t>(std::ceil((prefix_in_size + 1) / 8.f) * 8)];
	if (threadIdx.x == 0) {
		memcpy(prefix, prefix_in, prefix_in_size + 1);
		memcpy(miner_id_in, miner_id_in, miner_id_in_size + 1);
	}
	__syncthreads(); // Ensure the strings have been written to SMEM

	const uint64_t nonce = nonce_seed + (blockIdx.x * blockDim.x + threadIdx.x);

	// The first byte we can write because there is the input string at the begining
	// Respects the memory padding of 8 bit (uint8_t).
	const size_t minthreads_buffer
		= static_cast<size_t>(std::ceil((prefix_in_size + 1) / 8.f) * 8)
		+ static_cast<size_t>(std::ceil((miner_id_in_size + 1) / 8.f) * 8);

	const uintptr_t md_addr = threadIdx.x * (64) + minthreads_buffer;
	const uintptr_t nonce_addr = md_addr + 32;

	uint8_t* const md = &threads_buffer[md_addr];

	uint8_t* const coin_blob = &threads_buffer[nonce_addr];
	memset(coin_blob, 0, 32);

	uint8_t size = nonce_to_str(nonce, coin_blob);
	assert(size <= 32);
	{
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, prefix, prefix_in_size);
		sha256_update(&ctx, coin_blob, size);
		sha256_update(&ctx, miner_id_in, miner_id_in_size);
		sha256_final(&ctx, md);
	}

	if (count_leading_zero_nibbles_(md, difficulty) >= difficulty && atomicExch(out_found, 1) == 0) {
		memcpy(out_found_hash, md, 32);
		memcpy(out_input_string_nonce, prefix, prefix_in_size);
		memcpy(out_input_string_nonce + prefix_in_size, coin_blob, size);
		memcpy(out_input_string_nonce + prefix_in_size + size, miner_id_in, miner_id_in_size + 1);
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

	if (*g_found) {
		std::clog << g_out << std::endl;
		print_hash(g_hash_out);
	}
}


int main(const int argc, char const *const argv[]) {
	cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	t_last_updated = std::chrono::high_resolution_clock::now();

	const std::string arg_id_of_miner(argv[1]);
	// team_member_id // TODO
	const std::string last_coin(argv[3]);
	difficulty = std::stoi(argv[4]);
	// num_threads (ignored)

	const std::string prefix = std::string("CPEN 442 Coin2021") + last_coin;

	std::clog << "Nonce : ";
	std::cin >> user_nonce;

	char* g_prefix_str = nullptr;
	cudaMalloc(&g_prefix_str, prefix.size()+1);
	cudaMemcpy(g_prefix_str, prefix.c_str(), prefix.size()+1, cudaMemcpyHostToDevice);

	char* g_miner_id_str = nullptr;
	cudaMalloc(&g_miner_id_str, arg_id_of_miner.size()+1);
	cudaMemcpy(g_miner_id_str, arg_id_of_miner.c_str(), arg_id_of_miner.size()+1, cudaMemcpyHostToDevice);

	cudaMallocManaged(&g_out, prefix.size() + 32 + arg_id_of_miner.size() + 1);
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
			g_out, g_hash_out, g_found, g_prefix_str, prefix.size(), difficulty, nonce, g_miner_id_str, arg_id_of_miner.size()
		);
		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error("Device error");
		}
		nonce += NUMBLOCKS * BLOCK_SIZE;
		print_state();
	}

	cudaFree(g_out);
	cudaFree(g_hash_out);
	cudaFree(g_found);
	cudaFree(g_prefix_str);
	cudaFree(g_miner_id_str);
	cudaDeviceReset();
	return 0;
}