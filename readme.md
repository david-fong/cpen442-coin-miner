# CPEN 442 Coin Miner

https://blogs.ubc.ca/cpen442/coin-mining-contest/

## Usage

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# note: do this again every time the CMakeLists.txt files are changed

cmake --build build --config Release
# note: do this again every time the c++ files are changed

# run the miner:
python3 src/miner.py 96c3dfa5a7e119a8786e0b2c6dd332cf7302248611a1c37ffc6c2727e3a295b7 your_first_name_here num_threads
```

## Windows Environment Setup via WSL

Note: This assumes that you are setting up WSL for the first time.

- [install wsl](https://docs.microsoft.com/en-us/windows/wsl/install)

```
sudo apt install cmake
sudo apt install make
sudo apt-get install libssl-dev
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

- [in wsl, create an ssh key for git](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- [clone the repo](https://github.com/david-fong/cpen442-coin-miner)
- cd into the repo directory and follow the [Usage](#Usage) instructions
- (optional) Install the wsl extension for VS Code