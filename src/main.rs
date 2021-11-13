use std::fmt;
use sha2::{Sha256, Digest};
use base64;
mod miner;

fn main() {
   let last_coin = "a9c1ae3f4fc29d0be9113a42090a5ef9fdef93f5ec4777a008873972e60bb532";
   let id_of_miner_private_ = b"53976338";
   let id_of_miner_: String;
   {
      let mut sha = Sha256::default();
      sha.update(id_of_miner_private_);
      id_of_miner_ = fmt::format(format_args!("{:x}", sha.finalize()));
      assert_eq!(id_of_miner_.as_bytes().len(), 64);
      // I vehemently despise this hex string usage, but I don't write the rules.
      println!("id_of_miner: {:?}", id_of_miner_);
   }
   {
      let sha_prefix_ = miner::build_sha_prefix(last_coin);
      let coin_blob = miner::mine_coin(miner::MinerParams {
         sha_prefix: &sha_prefix_,
         difficulty: 8,
         id_of_miner: &id_of_miner_,
      });
      let coin_blob_base64 = base64::encode(&coin_blob);
      println!("success coin_blob: {:x}", coin_blob);
      println!("success coin_blob base64: {:#?}", coin_blob_base64);


      let mut sha = Sha256::default();
      sha.update(b"CPEN 442 Coin2021");
      sha.update(&last_coin);
      sha.update(&coin_blob);
      sha.update(&id_of_miner_);
      let test_result = sha.finalize();
      println!("test result: {:x}", test_result);
   }
}