import http.client
import json
import base64


class ChallengeParams:
	last_coin: str
	difficulty: int

	def __init__(self, last_coin: str, difficulty: int):
		self.last_coin = last_coin
		self.difficulty = difficulty


class Bank:
	"""
	note that connections are remade frequently because the bank server
	set a very short keep-alive value (5 seconds).
	"""
	url: str
	common_headers = {
		"Connection": "Keep-Alive",
		"Content-Type": "application/json",
	}

	def __init__(self, url: str):
		self.url = url


	def fetch_challenge(self, prev_fetched: ChallengeParams or None) -> ChallengeParams:
		conn = http.client.HTTPConnection(self.url)

		conn.request("POST", "/last_coin", headers=self.common_headers)
		res = conn.getresponse()
		j = json.loads(res.read())
		new_last_coin = j["coin_id"]
		if prev_fetched and new_last_coin == prev_fetched.last_coin:
			return prev_fetched

		conn.request("POST", "/difficulty", headers=self.common_headers)
		res = conn.getresponse()
		j = json.loads(res.read())
		return ChallengeParams(new_last_coin, j["number_of_leading_zeros"]) # *sad american spelling noises


	def claim_coin(self, id_of_miner, coin_blob_str):
		req_body = json.dumps({
			"coin_blob": base64.b64encode(bytes.fromhex(coin_blob_str)),
			"id_of_miner": id_of_miner,
		})
		conn = http.client.HTTPConnection(self.url)
		conn.request("POST", "/claim_coin", headers=self.common_headers, body=req_body)
		res = conn.getresponse()