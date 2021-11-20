add_executable(miner
   main.cpp
   miner.cpp
)
target_link_libraries(miner Threads::Threads)
target_link_libraries(miner OpenSSL::SSL)