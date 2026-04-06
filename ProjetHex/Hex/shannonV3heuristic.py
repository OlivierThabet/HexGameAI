from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse.linalg import cg

from game_state_hex import GameStateHex
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction


DIRS = ((0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1))
BRIDGE_TEMPLATES = (
    (1, -2, 0, -1, 1, -1),
    (2, -1, 1, -1, 1, 0),
    (1, 1, 0, 1, 1, 0),
    (-1, 2, 0, 1, -1, 1),
    (-2, 1, -1, 0, -1, 1),
    (-1, -1, -1, 0, 0, -1),
)
_GEOM_CACHE: Dict[int, dict] = {}


def _other(piece: str) -> str:
    return "B" if piece == "R" else "R"


def _get_geom(n: int) -> dict:
    if n in _GEOM_CACHE:
        return _GEOM_CACHE[n]

    size = n * n
    neighbors = [[] for _ in range(size)]
    bridge_links = [set() for _ in range(size)]
    rows = [0] * size
    cols = [0] * size

    for r in range(n):
        for c in range(n):
            idx = r * n + c
            rows[idx] = r
            cols[idx] = c

            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    neighbors[idx].append(nr * n + nc)

            for pdr, pdc, c1dr, c1dc, c2dr, c2dc in BRIDGE_TEMPLATES:
                pr, pc = r + pdr, c + pdc
                c1r, c1c = r + c1dr, c + c1dc
                c2r, c2c = r + c2dr, c + c2dc
                if not (
                    0 <= pr < n
                    and 0 <= pc < n
                    and 0 <= c1r < n
                    and 0 <= c1c < n
                    and 0 <= c2r < n
                    and 0 <= c2c < n
                ):
                    continue
                bridge_links[idx].add((pr * n + pc, c1r * n + c1c, c2r * n + c2c))

    geom = {
        "n": n,
        "size": size,
        "neighbors": neighbors,
        "bridge_links": [list(entries) for entries in bridge_links],
        "rows": rows,
        "cols": cols,
        "center": (n - 1) / 2.0,
    }
    _GEOM_CACHE[n] = geom
    return geom


def _has_won(board: List[str], piece: str, geom: dict) -> bool:
    visited = bytearray(len(board))
    stack: List[int] = []
    neighbors = geom["neighbors"]
    rows = geom["rows"]
    cols = geom["cols"]
    n = geom["n"]

    if piece == "R":
        for idx in range(len(board)):
            if rows[idx] == 0 and board[idx] == piece:
                visited[idx] = 1
                stack.append(idx)
        while stack:
            cur = stack.pop()
            if rows[cur] == n - 1:
                return True
            for nb in neighbors[cur]:
                if not visited[nb] and board[nb] == piece:
                    visited[nb] = 1
                    stack.append(nb)
    else:
        for idx in range(len(board)):
            if cols[idx] == 0 and board[idx] == piece:
                visited[idx] = 1
                stack.append(idx)
        while stack:
            cur = stack.pop()
            if cols[cur] == n - 1:
                return True
            for nb in neighbors[cur]:
                if not visited[nb] and board[nb] == piece:
                    visited[nb] = 1
                    stack.append(nb)
    return False


class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    MAX_EVAL_CACHE = 50000

    def __init__(self, piece_type: str, name: str = "AmperesBotV3") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict[Tuple[str, str, Tuple[str, ...]], float] = OrderedDict()
        self._eval_cache: OrderedDict[Tuple[str, Tuple[str, ...]], float] = OrderedDict()

    def compute_action(
        self,
        current_state: GameStateHex,
        remaining_time: float = 15 * 60,
        **kwargs,
    ) -> Action:
        del remaining_time, kwargs
        n = current_state.get_rep().get_dimensions()[0]
        geom = _get_geom(n)
        board, empties = self._extract_board(current_state, n)
        my_piece = self.piece_type
        opp_piece = _other(my_piece)

        if not empties:
            return self._to_action(0, n)

        # Opening book
        opening = self._opening_move(board, empties, my_piece, geom)
        if opening is not None:
            return self._to_action(opening, n)

        # Immediate win
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            move = self._ordered_moves(board, my_wins, my_piece, geom)[0]
            return self._to_action(move, n)

        # Block opponent win
        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            move = self._ordered_moves(board, opp_wins, my_piece, geom)[0]
            return self._to_action(move, n)

        # Greedy: pick the move with the best heuristic evaluation
        best_move = empties[0]
        best_score = float("-inf")

        for mv in self._ordered_moves(board, empties, my_piece, geom):
            board[mv] = my_piece
            if _has_won(board, my_piece, geom):
                board[mv] = "."
                return self._to_action(mv, n)
            score = self._evaluate_board(board, my_piece, geom)
            board[mv] = "."

            if score > best_score:
                best_score = score
                best_move = mv

        return self._to_action(best_move, n)

    def _evaluate_board(self, board: List[str], root_piece: str, geom: dict) -> float:
        board_key = tuple(board)
        cache_key = (root_piece, board_key)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            self._eval_cache.move_to_end(cache_key)
            return cached

        opp_piece = _other(root_piece)
        score = self._calculate_amperage(board_key, root_piece, geom) - self._calculate_amperage(
            board_key, opp_piece, geom
        )

        self._eval_cache[cache_key] = score
        if len(self._eval_cache) > self.MAX_EVAL_CACHE:
            self._eval_cache.popitem(last=False)
        return score

    def _calculate_amperage(self, board_key: Tuple[str, ...], target_piece: str, geom: dict) -> float:
        direction = "HAUTBAS" if target_piece == "R" else "GAUCHEDROITE"
        cache_key = (target_piece, direction, board_key)
        cached = self._amperage_cache.get(cache_key)
        if cached is not None:
            self._amperage_cache.move_to_end(cache_key)
            return cached

        n = geom["n"]
        size = geom["size"]
        g_mat = np.zeros((size, size))
        current = np.zeros(size)
        resistances = np.ones(size)

        for idx, cell in enumerate(board_key):
            if cell == target_piece:
                resistances[idx] = 0.001
            elif cell == ".":
                resistances[idx] = 1.0
            else:
                resistances[idx] = float("inf")

        for idx in range(size):
            r_i = resistances[idx]
            if r_i == float("inf"):
                g_mat[idx, idx] = 1.0
                continue

            diag_sum = 0.0
            for nb in geom["neighbors"][idx]:
                r_j = resistances[nb]
                if r_j == float("inf"):
                    continue
                conductance = 2.0 / (r_i + r_j)
                g_mat[idx, nb] -= conductance
                diag_sum += conductance

            source_c = 0.0
            sink_c = 0.0
            row = geom["rows"][idx]
            col = geom["cols"][idx]
            if direction == "HAUTBAS":
                if row == 0:
                    source_c = 2.0 / r_i
                if row == n - 1:
                    sink_c = 2.0 / r_i
            else:
                if col == 0:
                    source_c = 2.0 / r_i
                if col == n - 1:
                    sink_c = 2.0 / r_i

            g_mat[idx, idx] = diag_sum + source_c + sink_c
            current[idx] = source_c

        try:
            voltages, _ = cg(g_mat, current, rtol=1e-5)
            total_current = 0.0
            for i in range(n):
                row = 0 if direction == "HAUTBAS" else i
                col = i if direction == "HAUTBAS" else 0
                idx = row * n + col
                r_i = resistances[idx]
                if r_i == float("inf"):
                    continue
                source_c = 2.0 / r_i
                total_current += source_c * (1.0 - voltages[idx])
            result = float(total_current)
        except np.linalg.LinAlgError:
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    def _ordered_moves(self, board: List[str], moves: List[int], piece: str, geom: dict) -> List[int]:
        opp = _other(piece)
        rescue_moves = self._bridge_responses(board, piece, geom)
        cut_moves = self._bridge_cut_targets(board, opp, geom)
        scored = []

        for idx in moves:
            own_n = 0
            opp_n = 0
            for nb in geom["neighbors"][idx]:
                cell = board[nb]
                if cell == piece:
                    own_n += 1
                elif cell == opp:
                    opp_n += 1

            row = geom["rows"][idx]
            col = geom["cols"][idx]
            center = geom["center"]
            center_bias = -(abs(row - center) + abs(col - center))
            progress = -abs(col - center) if piece == "R" else -abs(row - center)

            bridge_bonus = self._bridge_build_bonus(board, idx, piece, geom)
            if idx in rescue_moves:
                bridge_bonus += 18.0
            if idx in cut_moves:
                bridge_bonus += 5.0

            score = 3.2 * own_n + 2.8 * opp_n + 0.25 * center_bias + 0.30 * progress + bridge_bonus
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _bridge_responses(self, board: List[str], piece: str, geom: dict) -> Set[int]:
        opp = _other(piece)
        responses: Set[int] = set()

        for idx, cell in enumerate(board):
            if cell != piece:
                continue
            for partner, c1, c2 in geom["bridge_links"][idx]:
                if board[partner] != piece:
                    continue
                if board[c1] == opp and board[c2] == ".":
                    responses.add(c2)
                elif board[c2] == opp and board[c1] == ".":
                    responses.add(c1)

        return responses

    def _bridge_cut_targets(self, board: List[str], opp: str, geom: dict) -> Set[int]:
        targets: Set[int] = set()

        for idx, cell in enumerate(board):
            if cell != opp:
                continue
            for partner, c1, c2 in geom["bridge_links"][idx]:
                if board[partner] != opp:
                    continue
                if board[c1] == "." and board[c2] == ".":
                    targets.add(c1)
                    targets.add(c2)

        return targets

    def _bridge_build_bonus(self, board: List[str], idx: int, piece: str, geom: dict) -> float:
        opp = _other(piece)
        bonus = 0.0

        for partner, c1, c2 in geom["bridge_links"][idx]:
            if board[partner] != piece:
                continue
            v1 = board[c1]
            v2 = board[c2]
            if v1 == "." and v2 == ".":
                bonus += 3.0
            elif (v1 == piece and v2 == ".") or (v2 == piece and v1 == "."):
                bonus += 4.5
            elif (v1 == opp and v2 == ".") or (v2 == opp and v1 == "."):
                bonus += 0.5

        return bonus

    def _winning_moves(self, board: List[str], empties: List[int], piece: str, geom: dict) -> List[int]:
        wins = []
        for idx in empties:
            board[idx] = piece
            won = _has_won(board, piece, geom)
            board[idx] = "."
            if won:
                wins.append(idx)
        return wins

    def _opening_move(self, board: List[str], empties: List[int], piece: str, geom: dict) -> int | None:
        stones_played = len(board) - len(empties)
        if stones_played >= 4:
            return None

        n = geom["n"]
        center = n // 2
        candidates = [
            center * n + center,
            center * n + max(0, center - 1),
            center * n + min(n - 1, center + 1),
            max(0, center - 1) * n + center,
            min(n - 1, center + 1) * n + center,
        ]
        available = [idx for idx in candidates if 0 <= idx < n * n and board[idx] == "."]
        if not available:
            return None
        return self._ordered_moves(board, available, piece, geom)[0]

    def _extract_board(self, state: GameStateHex, n: int) -> Tuple[List[str], List[int]]:
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [idx for idx, cell in enumerate(board) if cell == "."]
        return board, empties

    def _to_action(self, idx: int, n: int) -> StatelessAction:
        return StatelessAction({"piece": self.piece_type, "position": (idx // n, idx % n)})