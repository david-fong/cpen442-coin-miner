import subprocess
import os
import sys
import platform
import time

import bank


POLLING_PERIOD_SECONDS = (60/10)+1


class MinerParams:
	bank_url: str
	id_of_miner: str
	team_member_id: str
	num_threads: int

	def __init__(self, bank_url: str or None, id_of_miner: str, team_member_id: str, num_threads: int) -> None:
		self.bank_url = bank_url or "cpen442coin.ece.ubc.ca"
		self.id_of_miner = id_of_miner
		self.team_member_id = team_member_id
		self.num_threads = num_threads


def setup_new_mine(bonk: bank.Bank, clear_drop_cache=True) -> bank.ChallengeParams:
	if clear_drop_cache:
		try:
			os.remove(os.path.join(os.getcwd(), "wallet", "difficulty_drop_cache.txt"))
		except:
			pass

	challenge = bonk.fetch_challenge()
	while challenge == None:
		time.sleep(POLLING_PERIOD_SECONDS)
		challenge = bonk.fetch_challenge()
	return challenge


def start_mining(params: MinerParams):
	miner_exe = {
		"Linux":   os.path.join(os.getcwd(), "build", "src", "miner"),
		"Darwin":  os.path.join(os.getcwd(), "build", "src", "miner"),
		"Windows": os.path.join(os.getcwd(), "build", "src", "Release", "miner.exe"),
	}[platform.system()]
	print(miner_exe)
	bonk = bank.Bank(params.bank_url)

	challenge = setup_new_mine(bonk, clear_drop_cache=False)
	while True:
		miners_proc = subprocess.Popen(
			[miner_exe,
			params.id_of_miner, params.team_member_id,
			challenge.last_coin, str(challenge.difficulty),
			str(params.num_threads)],
			stdout=subprocess.PIPE,
			encoding="utf-8", text=True,
		)
		while True:
			try:
				stdout, stderr = miners_proc.communicate(timeout=POLLING_PERIOD_SECONDS)
				# print(stderr)
				print("🎊 found a coin!\n")
				bonk.claim_coin(id_of_miner=params.id_of_miner, coin_blob_str=stdout)
				challenge = setup_new_mine(bonk)
				break
			except subprocess.TimeoutExpired:
				new_chl = bonk.fetch_challenge()
				if (new_chl != None):
					is_new_last_coin = new_chl.last_coin != challenge.last_coin
					is_new_difficulty = new_chl.difficulty != challenge.difficulty
					if is_new_difficulty and not is_new_last_coin:
						# try to use the difficulty_drop_cache
						cache_path = os.path.join(os.getcwd(), "wallet", "difficulty_drop_cache.txt")
						if os.path.isfile(cache_path):
							try:
								with open(cache_path, mode="r") as cache:
									cache_difficulty, coin_blob_str = str.split(cache.readline())
									if (int(cache_difficulty) >= new_chl.difficulty):
										print("🎊 found a difficulty_drop_cache coin!\n")
										bonk.claim_coin(id_of_miner=params.id_of_miner, coin_blob_str=coin_blob_str)
										challenge = setup_new_mine(bonk)
										break
							except Exception as err:
								print("error reading difficulty_drop_cache:\n" + err)

					if is_new_last_coin or is_new_difficulty:
						miners_proc.kill()
						challenge = new_chl
						break
				# else continue polling for miners_proc finish
			except (KeyboardInterrupt, Exception):
				# catch-all clause to kill the miner program
				miners_proc.kill()
				raise


def main() -> int:
	start_mining(MinerParams(None, sys.argv[1], sys.argv[2], int(sys.argv[3])))
	return 0


if __name__ == "__main__":
	sys.exit(main())