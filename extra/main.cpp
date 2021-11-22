#include <pcap/pcap.h>

#include "miner.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * argv[1] device
 */
int main(const int argc, char const *const argv[]) {
	if (argc < 4) {
		std::cerr << "missing args. check main.cpp for args format.";
		return 1;
	}

	char* dev;
	char errbuf[PCAP_ERRBUF_SIZE];
	dev = pcap_lookupdev(errbuf);
	if (dev == NULL) {

	}
	return 0;
}