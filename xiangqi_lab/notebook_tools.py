"""Small, standard-library helpers for the environment notebook (no trainer)."""
from html import escape

from rules import Game, PRESETS


ENDGAMES = {p["id"]: dict(p) for p in PRESETS if p["id"] in ("rook", "cannon", "horse")}
PIECE_LABELS = dict(zip("KAEHRCPkaehrcp", "帥仕相馬車炮兵將士象馬車砲卒"))


def action_mask(env):
    """An 8100-entry mask; an ended episode has no available actions."""
    if env.game is None:
        raise RuntimeError("Call reset before asking for legal actions.")
    mask = [False] * 8100
    if env.game.outcome() is None:
        for index in env.game.observation()["legal_action_ids"]:
            mask[index] = True
    return mask


def replay_episode(episode):
    """Replay the website's JSON format, validating every move and final FEN."""
    if not isinstance(episode, dict) or episode.get("schema") != "xiangqi-lab.episode.v1":
        raise ValueError("Unsupported episode format.")
    moves = episode.get("moves", [])
    limit = episode.get("max_plies", 160)
    if not isinstance(moves, list) or len(moves) > 1000:
        raise ValueError("Invalid move list.")
    if type(limit) is not int or not 2 <= limit <= 1000:
        raise ValueError("The ply limit must be between 2 and 1000.")
    game = Game(episode.get("initial_fen", ""), limit)
    for entry in moves:
        if not isinstance(entry, dict):
            raise ValueError("Invalid move record.")
        game.step(entry.get("move", ""), actor=str(entry.get("actor", "import"))[:100])
    if episode.get("final_fen") and game.fen != episode["final_fen"]:
        raise ValueError("The move history does not match the final FEN.")
    return game


def board_svg(game, title="Xiangqi Lab"):
    """Browser-rendered SVG: Red at the bottom, traditional piece symbols."""
    title = escape(str(title))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 744" '
        'width="620" role="img" aria-label="Xiangqi board">',
        f'<title>{title}</title>',
        '<rect width="620" height="744" rx="18" fill="#f6f1e8"/>',
        '<g font-family="Arial, sans-serif" fill="#29423e">',
        f'<text x="40" y="38" font-size="22" font-weight="bold">{title}</text>',
        f'<text x="40" y="65" font-size="15">Turn: {game.side.title()} · Move count: {len(game.history)}</text>',
        '</g>',
        '<rect x="32" y="85" width="556" height="612" rx="10" fill="#efe1c7"/>',
        '<g stroke="#a68b61" stroke-width="1.4">',
    ]

    def line(x1, y1, x2, y2):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')

    for row in range(10):
        line(70, 120 + row * 60, 550, 120 + row * 60)
    for col in range(9):
        x = 70 + col * 60
        if col in (0, 8):
            line(x, 120, x, 660)
        else:
            line(x, 120, x, 360)
            line(x, 420, x, 660)
    for y in (120, 540):
        line(250, y, 370, y + 120)
        line(370, y, 250, y + 120)
    parts += ['</g>', '<g font-family="Arial, sans-serif" text-anchor="middle" fill="#806a48">',
              '<text x="310" y="397" font-size="18" letter-spacing="6">RIVER</text>']
    for col in range(9):
        parts.append(f'<text x="{70 + col * 60}" y="688" font-size="13">{chr(97 + col)}</text>')
    for row in range(10):
        parts.append(f'<text x="20" y="{125 + row * 60}" font-size="13">{9 - row}</text>')
    parts.append('</g>')
    for row, cells in enumerate(game.board):
        for col, piece in enumerate(cells):
            if not piece:
                continue
            x, y = 70 + col * 60, 120 + row * 60
            ink = "#ab3e2f" if piece.isupper() else "#29423e"
            parts.extend([
                f'<g><title>{piece} on {chr(97 + col)}{9 - row}</title>',
                f'<circle cx="{x}" cy="{y}" r="25" fill="#fff6dc" stroke="{ink}" stroke-width="1.4"/>',
                f'<circle cx="{x}" cy="{y}" r="21.5" fill="none" stroke="#d7c49e"/>',
                f'<text x="{x}" y="{y + 11}" text-anchor="middle" font-size="32" '
                f'font-family="Noto Serif CJK SC, Microsoft YaHei, PingFang SC, serif" fill="{ink}">{PIECE_LABELS[piece]}</text></g>',
            ])
    result = game.outcome()
    footer = "Game in progress" if result is None else f"End: {result['reason']} · Winner: {result['winner'] or 'none'}"
    parts.append(f'<text x="40" y="726" font-size="15" font-family="Arial, sans-serif" fill="#29423e">{escape(footer)}</text></svg>')
    return "\n".join(parts)
