use sha2::{Sha256, Digest, digest::{generic_array::{GenericArray, typenum}}};

const CHALLENGE_PREFIX: &[u8; 17] = b"CPEN 442 Coin2021";
const COIN_BLOB_BYTES: usize = 38; // 64 - 17 - 9
type CoinBlobBytes = typenum::U38;

pub struct MinerParams<'a> {
   /// nibbles
   pub difficulty: u32,
   pub sha_prefix: &'a Sha256,
   /// I vehemently despise this hex string usage, but I don't write the rules.
   pub id_of_miner: &'a String,
}

pub fn build_sha_prefix(last_coin: &str) -> Sha256 {
   let mut sha = Sha256::default();
   sha.update(CHALLENGE_PREFIX);
   sha.update(last_coin);
   return sha;
}

fn permute_coin_blob(coin_blob: &mut GenericArray<u8, CoinBlobBytes>) {
   for i in 0..COIN_BLOB_BYTES {
      if coin_blob[i] == 255u8 {
         coin_blob[i] = 0;
      } else {
         coin_blob[i] += 1;
         break;
      }
   }
}

fn check_miner_success(difficulty: u32, digest: &[u8]) -> bool {
   let bytes = (difficulty / 2) as usize;
   for i in 0..bytes { if digest[i] != 0 { return false; } }
   if (difficulty % 2 == 1) && (digest[bytes] & 0xF0u8 != 0) { return false; }
   return true;
}

pub fn mine_coin(params: MinerParams) -> GenericArray<u8, CoinBlobBytes> {
   let mut coin_blob = GenericArray::<u8, CoinBlobBytes>::default();
   coin_blob[3] = 0xea; // TODO delete when doing the coin mining contest
   coin_blob[4] = 0x01; // TODO delete when doing the coin mining contest
   loop {
      if check_miner_success(6, &coin_blob) {
         println!("coin_blob progress: {:x}", coin_blob);
      }
      let mut sha: Sha256 = params.sha_prefix.clone();
      sha.update(&coin_blob);
      sha.update(&params.id_of_miner);
      let digest = sha.finalize();

      if check_miner_success(params.difficulty, &digest) {
         println!("success digest: {:x}", digest);
         return coin_blob;
      }
      permute_coin_blob(&mut coin_blob);
   }
}