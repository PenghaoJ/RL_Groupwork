"""Turn-based two-player environment; no ML framework required.

reset -> (observation, info); step -> (observation, reward, terminated, truncated, info)
Reward is from the ACTING side's perspective, not always Red's.
This API resembles Gymnasium but is not a gymnasium.Env subclass.
"""
from rules import Game, START_FEN, action_move

class XiangqiEnv:
    def __init__(self, max_plies=160):
        self.max_plies=max_plies
        self.game=None

    def reset(self, *, fen=START_FEN, seed=None):
        self.game=Game(fen,self.max_plies)
        return self.game.observation(), {"outcome":self.game.outcome(),"seed":seed}

    def step(self, action):
        if self.game is None: raise RuntimeError("Call reset before step.")
        move=action_move(action) if type(action) is int else action
        side=self.game.side
        result=self.game.step(move,actor="training")
        reward=0.0
        if result and result["winner"]:
            reward=1.0 if result["winner"] == side else -1.0
        return self.game.observation(), reward, bool(result and result["terminated"]), bool(result and result["truncated"]), {"acting_side":side,"outcome":result,"rewards":{"red": (1 if result and result["winner"] == "red" else -1 if result and result["winner"] == "black" else 0),"black": (1 if result and result["winner"] == "black" else -1 if result and result["winner"] == "red" else 0)}}
