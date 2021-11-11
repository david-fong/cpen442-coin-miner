use sha2::{Sha256, Digest, digest::{generic_array::{GenericArray, typenum, sequence::Concat}}};

const CHALLENGE_PREFIX: &[u8; 17] = b"CPEN 442 Coin2021";
type PrefixBlockLeftoverBytes = typenum::U15; // 64 - 17 - 32
const COIN_BLOB_BLOCK_BYTES: usize = 23; // 64 - 32 - 8 - 1
type CoinBlobBlockBytes = typenum::U23;
type CoinBlobBytes = typenum::U38; // + PrefixBlockLeftoverBytes

pub struct MinerParams<'a> {
   /// nibbles
   pub difficulty: u32,
   pub sha_prefix: &'a Sha256,
   pub id_of_miner: &'a GenericArray<u8, typenum::U32>,
}

pub fn build_sha_prefix(last_coin: &str) -> Sha256 {
   let mut block: [u8; 64] = [0; 64];
   block[0..CHALLENGE_PREFIX.len()].copy_from_slice(CHALLENGE_PREFIX);
   {
      let offset = CHALLENGE_PREFIX.len();
      hex::decode_to_slice(last_coin, &mut block[offset..offset+32]).unwrap();
   }
   let a: GenericArray<u8, typenum::U64> = GenericArray::clone_from_slice(&block[0..64]);
   println!("prefix block: {:x}", a);
   let mut sha = Sha256::default();
   sha.update(block);
   return sha;
}

fn permute_coin_blob(coin_blob: &mut GenericArray<u8, CoinBlobBlockBytes>) {
   for i in 0..COIN_BLOB_BLOCK_BYTES {
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

   let mut coin_blob = GenericArray::<u8, CoinBlobBlockBytes>::default();
   coin_blob[3] = 24; // TODO delete when doing the coin mining contest
   loop {
      if check_miner_success(6, &coin_blob) {
         println!("coin_blob progress: {:x}", coin_blob);
      }
      let mut sha: Sha256 = params.sha_prefix.clone();
      sha.update(coin_blob);
      sha.update(params.id_of_miner);
      let digest = sha.finalize();

      if check_miner_success(params.difficulty, &digest) {
         println!("success digest: {:x}", digest);
         return GenericArray::<u8, PrefixBlockLeftoverBytes>::default().concat(coin_blob);
      }
      permute_coin_blob(&mut coin_blob);
   }
}