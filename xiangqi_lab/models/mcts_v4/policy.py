"""Route received v4 memory from the game start, including Black and shared workers."""
from received_policy import Policy as ReceivedPolicy
from rules import PRESETS


class Policy(ReceivedPolicy):
    def _identify_scene(self, observation):
        # A worker serves multiple games. Never infer a new game's scene from
        # the previous request, or from its changing material during play.
        initial_fen = observation.get("initial_fen")
        if initial_fen is None and observation.get("ply") == 0:
            initial_fen = observation["fen"]
        key = " ".join((initial_fen or "").split()[:2])
        starts = [" ".join(p["fen"].split()[:2]) for p in PRESETS[:4]]
        scene = starts.index(key) if key in starts else None
        if scene != self.scene_id:
            self._experience_mtime = None
        self.scene_id = scene
        self._last_ply = observation.get("ply", 0)
