"""Rotate the board and exchange colors for a policy trained only as Red."""

PIECES = "KAEHRCPkaehrcp"


def flip_action(action):
    if type(action) is not int or not 0 <= action < 8100:
        raise ValueError("Expected an action ID from 0 to 8099.")
    source, destination = divmod(action, 90)
    return (89 - source) * 90 + 89 - destination


def flip_move(move):
    if len(move) != 4 or any(move[i] not in "abcdefghi" for i in (0, 2)) or any(move[i] not in "0123456789" for i in (1, 3)):
        raise ValueError("Expected a coordinate move such as a3a4.")
    return "".join(chr(ord("i") - (ord(move[i]) - ord("a"))) + str(9 - int(move[i + 1])) for i in (0, 2))


def flip_fen(fen):
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in ("w", "b", "r"):
        raise ValueError("Expected a FEN position and side.")
    ranks = parts[0].split("/")
    if len(ranks) != 10:
        raise ValueError("Expected ten board ranks.")
    parts[0] = "/".join(rank[::-1].swapcase() for rank in reversed(ranks))
    parts[1] = "w" if parts[1] == "b" else "b"
    return " ".join(parts)


def flip_observation(observation):
    if observation["plane_order"] != PIECES:
        raise ValueError("Unsupported piece plane order.")
    if observation["side_to_move"] not in ("red", "black"):
        raise ValueError("Unknown side to move.")
    planes = observation["board_planes"]
    return {
        **observation,
        "fen": flip_fen(observation["fen"]),
        "side_to_move": "red" if observation["side_to_move"] == "black" else "black",
        "board_planes": [[list(reversed(row)) for row in reversed(planes[(i + 7) % 14])] for i in range(14)],
        "legal_moves": [flip_move(move) for move in observation["legal_moves"]],
        "legal_action_ids": [flip_action(action) for action in observation["legal_action_ids"]],
        "repetition_counts": {flip_fen(key): count for key, count in observation["repetition_counts"].items()},
    }


def as_red(observation):
    if observation["side_to_move"] == "red":
        return observation, False
    if observation["side_to_move"] != "black":
        raise ValueError("Unknown side to move.")
    return flip_observation(observation), True
