import time
import http.client
import json
import base64


POLLING_PERIOD_SECONDS = (60/10)+1


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


	def _fetch_challenge(self) -> ChallengeParams or None:
		"""Returns None if the http request failed"""
		ret = None
		try:
			conn = http.client.HTTPConnection(self.url)

			conn.request("POST", "/last_coin", headers=self.common_headers)
			res = conn.getresponse()
			j = json.loads(res.read())
			new_last_coin = j["coin_id"]

			conn.request("POST", "/difficulty", headers=self.common_headers)
			res = conn.getresponse()
			j = json.loads(res.read())
			ret = ChallengeParams(new_last_coin, j["number_of_leading_zeros"]) # *sad american spelling noises

		except KeyboardInterrupt as err:
			raise err
		except Exception as err:
			print(err)
		finally:
			return ret


	def fetch_challenge(self) -> ChallengeParams:
		"""Polls according to POLLING_PERIOD_SECONDS upon failed http requests."""
		challenge = self._fetch_challenge()
		while challenge == None:
			time.sleep(POLLING_PERIOD_SECONDS)
			challenge = self._fetch_challenge()
		return challenge


	def claim_coin(self, id_of_miner: str, coin_blob_str: str):
		coin_blob_str_base64 = str(base64.b64encode(bytes.fromhex(coin_blob_str)), "ascii")
		try:
			req_body = json.dumps({
				"coin_blob": coin_blob_str_base64,
				"id_of_miner": id_of_miner,
			})
			conn = http.client.HTTPConnection(self.url)
			conn.request("POST", "/claim_coin", headers=self.common_headers, body=req_body)
			res = conn.getresponse()

		except KeyboardInterrupt as err:
			raise err
		except Exception as err:
			print(err)
			print("🚨 http request to claim coin failed! please claim it manually")
			print("coin_blob base64: " + coin_blob_str_base64 + "\n")