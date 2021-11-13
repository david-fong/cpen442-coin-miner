add_executable(miner
   main.cpp
   miner.cpp
   net_api.cpp
)
target_link_libraries(miner OpenSSL::SSL)