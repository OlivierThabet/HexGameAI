from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

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

# Fraction of remaining time we're willing to spend on one move
TIME_BUDGET_FRACTION = 0.08
# UCB1 exploration constant — higher = more exploration
UCB_C = 1.4


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


# ─────────────────────────────────────────────
#  MCTS Node
# ─────────────────────────────────────────────

class _Node:
    """A node in the MCTS tree."""

    __slots__ = (
        "move",       # move (idx) that led to this node, None for root
        "to_play",    # piece that just played to reach this node
        "parent",
        "children",
        "untried",    # moves not yet expanded
        "visits",
        "value",      # cumulative score from root_piece perspective
    )

    def __init__(
        self,
        move: Optional[int],
        to_play: str,
        parent: Optional[_Node],
        untried: List[int],
    ) -> None:
        self.move = move
        self.to_play = to_play
        self.parent = parent
        self.children: List[_Node] = []
        self.untried = untried
        self.visits = 0
        self.value = 0.0

    def ucb(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + UCB_C * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def best_child(self) -> "_Node":
        return max(self.children, key=lambda c: c.ucb(self.visits))

    def most_visited_child(self) -> "_Node":
        return max(self.children, key=lambda c: c.visits)


# ─────────────────────────────────────────────
#  Player
# ─────────────────────────────────────────────

class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000

    def __init__(self, piece_type: str, name: str = "AmperesMCTS") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict = OrderedDict()

    # ── Public entry point ──────────────────────────────────────────────

    def compute_action(
        self,
        current_state: GameStateHex,
        remaining_time: float = 15 * 60,
        **kwargs,
    ) -> Action:
        del kwargs
        n = current_state.get_rep().get_dimensions()[0]
        geom = _get_geom(n)
        board, empties = self._extract_board(current_state, n)
        my_piece = self.piece_type
        opp_piece = _other(my_piece)

        if not empties:
            return self._to_action(0, n)

        opening = self._opening_move(board, empties, my_piece, geom)
        if opening is not None:
            return self._to_action(opening, n)

        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(my_wins[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(opp_wins[0], n)

        budget = remaining_time * TIME_BUDGET_FRACTION
        best_move = self._mcts(board, empties, my_piece, geom, budget)
        return self._to_action(best_move, n)

    # ── MCTS ────────────────────────────────────────────────────────────

    def _mcts(
        self,
        board: List[str],
        empties: List[int],
        root_piece: str,
        geom: dict,
        budget: float,
    ) -> int:
        ordered = self._relevant_ordered_moves(board, empties, root_piece, geom)
        root = _Node(move=None, to_play=root_piece, parent=None, untried=ordered)
        deadline = time.time() + budget

        while time.time() < deadline:
            # Selection
            node, sim_board, sim_empties = self._select(
                root, board[:], empties[:], geom
            )

            # Expansion — if this node has untried moves and game isn't over
            if node.untried and not _has_won(sim_board, _other(node.to_play), geom):
                move = node.untried.pop(0)
                sim_board[move] = node.to_play
                sim_empties = [e for e in sim_empties if e != move]
                child_moves = self._relevant_ordered_moves(
                    sim_board, sim_empties, _other(node.to_play), geom
                )
                child = _Node(
                    move=move,
                    to_play=_other(node.to_play),
                    parent=node,
                    untried=child_moves,
                )
                node.children.append(child)
                node = child

            # Simulation — amperage as leaf evaluation
            value = self._simulate(sim_board, root_piece, geom)

            # Backpropagation
            self._backprop(node, value)

        return root.most_visited_child().move

    def _select(
        self,
        node: _Node,
        board: List[str],
        empties: List[int],
        geom: dict,
    ) -> Tuple[_Node, List[str], List[int]]:
        """Walk down the tree via UCB1 until we reach an expandable node."""
        while not node.untried and node.children:
            node = node.best_child()
            board[node.move] = node.to_play
            empties = [e for e in empties if e != node.move]
            if _has_won(board, node.to_play, geom):
                break
        return node, board, empties

    def _simulate(self, board: List[str], root_piece: str, geom: dict) -> float:
        """
        Leaf evaluation using amperage.
        Returns a normalized score in (-1, 1) relative to root_piece.
        """
        if _has_won(board, root_piece, geom):
            return 1.0
        if _has_won(board, _other(root_piece), geom):
            return -1.0

        board_key = tuple(board)
        my_amp = self._calculate_amperage(board_key, root_piece, geom)
        opp_amp = self._calculate_amperage(board_key, _other(root_piece), geom)

        total = my_amp + opp_amp
        if total == 0:
            return 0.0
        return (my_amp - opp_amp) / total  # normalized to (-1, 1)

    def _backprop(self, node: _Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    # ── Move generation ─────────────────────────────────────────────────

    def _relevant_ordered_moves(
        self,
        board: List[str],
        empties: List[int],
        piece: str,
        geom: dict,
        radius: int = 2,
    ) -> List[int]:
        """Only keep moves within `radius` of an existing piece, then order them."""
        relevant = set()
        for idx, cell in enumerate(board):
            if cell == ".":
                continue
            r, c = geom["rows"][idx], geom["cols"][idx]
            for e in empties:
                er, ec = geom["rows"][e], geom["cols"][e]
                if abs(er - r) + abs(ec - c) <= radius:
                    relevant.add(e)
        candidates = list(relevant) if relevant else empties
        return self._ordered_moves(board, candidates, piece, geom)

    def _ordered_moves(
        self, board: List[str], moves: List[int], piece: str, geom: dict
    ) -> List[int]:
        opp = _other(piece)
        scored = []
        for idx in moves:
            own_n = sum(1 for nb in geom["neighbors"][idx] if board[nb] == piece)
            opp_n = sum(1 for nb in geom["neighbors"][idx] if board[nb] == opp)
            row, col = geom["rows"][idx], geom["cols"][idx]
            center = geom["center"]
            center_bias = -(abs(row - center) + abs(col - center))
            progress = -abs(col - center) if piece == "R" else -abs(row - center)
            score = 3.2 * own_n + 2.8 * opp_n + 0.25 * center_bias + 0.30 * progress
            scored.append((score, idx))
        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _winning_moves(
        self, board: List[str], empties: List[int], piece: str, geom: dict
    ) -> List[int]:
        wins = []
        for idx in empties:
            board[idx] = piece
            if _has_won(board, piece, geom):
                wins.append(idx)
            board[idx] = "."
        return wins

    def _opening_move(
        self, board: List[str], empties: List[int], piece: str, geom: dict
    ) -> Optional[int]:
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

    # ── Amperage ─────────────────────────────────────────────────────────

    def _calculate_amperage(
        self, board_key: Tuple[str, ...], target_piece: str, geom: dict
    ) -> float:
        direction = "HAUTBAS" if target_piece == "R" else "GAUCHEDROITE"
        cache_key = (target_piece, direction, board_key)
        cached = self._amperage_cache.get(cache_key)
        if cached is not None:
            self._amperage_cache.move_to_end(cache_key)
            return cached

        n = geom["n"]
        size = geom["size"]
        opponent_piece = _other(target_piece)

        def edge_conductance(cell_i: str, cell_j: str) -> float:
            if cell_i == opponent_piece or cell_j == opponent_piece:
                return 0.0
            if cell_i == target_piece and cell_j == target_piece:
                return 1000.0
            if cell_i == target_piece or cell_j == target_piece:
                return 2.0
            return 1.0

        def source_conductance(cell: str) -> float:
            if cell == opponent_piece:
                return 0.0
            if cell == target_piece:
                return 1000.0
            return 1.0

        g_mat = np.zeros((size, size))
        current = np.zeros(size)

        for idx in range(size):
            cell_i = board_key[idx]
            if cell_i == opponent_piece:
                g_mat[idx, idx] = 1.0
                continue

            diag_sum = 0.0

            for nb in geom["neighbors"][idx]:
                cell_j = board_key[nb]
                c = edge_conductance(cell_i, cell_j)
                if c == 0.0:
                    continue
                g_mat[idx, nb] -= c
                diag_sum += c

            for partner, c1, c2 in geom["bridge_links"][idx]:
                if (
                    cell_i == target_piece
                    and board_key[partner] == target_piece
                    and board_key[c1] == "."
                    and board_key[c2] == "."
                ):
                    c = 500.0
                    g_mat[idx, partner] -= c
                    diag_sum += c

            source_c = 0.0
            sink_c = 0.0
            row = geom["rows"][idx]
            col = geom["cols"][idx]
            if direction == "HAUTBAS":
                if row == 0:     source_c = source_conductance(cell_i)
                if row == n - 1: sink_c   = source_conductance(cell_i)
            else:
                if col == 0:     source_c = source_conductance(cell_i)
                if col == n - 1: sink_c   = source_conductance(cell_i)

            g_mat[idx, idx] = diag_sum + source_c + sink_c
            current[idx] = source_c

        try:
            voltages, _ = cg(g_mat, current, rtol=1e-5)
            total_current = 0.0
            for i in range(n):
                row = 0 if direction == "HAUTBAS" else i
                col = i if direction == "HAUTBAS" else 0
                idx = row * n + col
                cell = board_key[idx]
                if cell == opponent_piece:
                    continue
                sc = source_conductance(cell)
                total_current += sc * (1.0 - voltages[idx])
            result = float(total_current)
        except np.linalg.LinAlgError:
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_board(
        self, state: GameStateHex, n: int
    ) -> Tuple[List[str], List[int]]:
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [idx for idx, cell in enumerate(board) if cell == "."]
        return board, empties

    def _to_action(self, idx: int, n: int) -> StatelessAction:
        return StatelessAction(
            {"piece": self.piece_type, "position": (idx // n, idx % n)}
        )
