import subprocess
import os
import sys

import bank


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


def start_mining(params: MinerParams):
	bonk = bank.Bank(params.bank_url)
	challenge = bonk.fetch_challenge(None)
	while True:
		miners_proc = subprocess.Popen(
			[os.path.join(os.getcwd(), "build", "src", "miner"),
			params.id_of_miner, params.team_member_id,
			challenge.last_coin, str(challenge.difficulty),
			str(params.num_threads)],
			stdout=subprocess.PIPE,
			encoding="utf-8", text=True,
		)
		while True:
			try:
				stdout, = miners_proc.communicate(timeout=(60/10)+1)
				bonk.claim_coin(id_of_miner=params.id_of_miner, coin_blob=stdout)
				challenge = bonk.fetch_challenge(challenge)
				break
			except subprocess.TimeoutExpired:
				new_challenge = bonk.fetch_challenge(challenge)
				if new_challenge.last_coin != challenge.last_coin:
					miners_proc.kill()
					challenge = new_challenge
					break
				# else continue polling for miners_proc finish


def main() -> int:
	start_mining(MinerParams(None, sys.argv[1], sys.argv[2], int(sys.argv[3])))
	return 0


if __name__ == "__main__":
	sys.exit(main())