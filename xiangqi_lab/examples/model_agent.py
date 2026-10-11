"""Replace the policy here with inference from your own checkpoint.

Only emit the response JSON on stdout; diagnostic logs belong on stderr.
The packaged Python has no torch/numpy dependencies. Register your training
environment's Python executable in models.json when your model needs them.
"""
import json
import random
import sys

request=json.loads(sys.stdin.readline())
observation=request["observation"]
# Example only; this is not a trained neural model.
move=random.choice(observation["legal_moves"])
print(json.dumps({"move":move}))
