import subprocess
import os
import sys

import bank


class MinerParams:
	id_of_miner: str
	team_member_id: str
	num_threads: int
	def __init__(self) -> None:
		pass


def start_mining(params: MinerParams):
	completed_process = subprocess.Popen(
		[os.path.join("build", "src", "miner"), id_of_miner, team_member_id, last_coin, difficulty, num_threads],
		stdout=subprocess.PIPE,
		encoding="utf-8", text=True,
	)
	completed_process.stdout


def main() -> int:
	start_mining()
	return 0


if __name__ == "__main__":
	sys.exit(main())