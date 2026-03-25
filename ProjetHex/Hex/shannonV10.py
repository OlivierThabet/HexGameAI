'''
Agent hybride MCTS → MiniMax.
- Ouverture : premiers coups au centre
- Early/mid game (beaucoup de cases vides) : MCTS
- Late game (peu de cases vides) : MiniMax iterative deepening
- Pas d'active area
- Modèle électrique V3 + bridges alliés (symétriques)
- Ordonnancement V3
'''

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix
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
    bridge_links = [[] for _ in range(size)]
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

            seen = set()
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
                key = (pr * n + pc, c1r * n + c1c, c2r * n + c2c)
                if key not in seen:
                    seen.add(key)
                    bridge_links[idx].append(key)

    grad_row = np.array([1.0 - r / (n - 1) for r in range(n) for _ in range(n)])
    grad_col = np.array([1.0 - c / (n - 1) for _ in range(n) for c in range(n)])

    geom = {
        "n": n,
        "size": size,
        "neighbors": neighbors,
        "bridge_links": bridge_links,
        "rows": rows,
        "cols": cols,
        "center": (n - 1) / 2.0,
        "grad_row": grad_row,
        "grad_col": grad_col,
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


# ---------------------------------------------------------------------------
# Noeud MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("move", "parent", "children", "visits", "value", "untried", "to_play")

    def __init__(self, move, parent, untried, to_play):
        self.move = move
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried = untried
        self.to_play = to_play

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration: float = 1.414) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.visits)


# ---------------------------------------------------------------------------
# Agent hybride
# ---------------------------------------------------------------------------

class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    MAX_EVAL_CACHE = 50000
    CG_MAXITER = 40
    TIME_FRACTION = 0.06
    MIN_TIME_PER_MOVE = 0.5
    EXPLORATION = 1.414
    ROLLOUT_DEPTH = 6
    MINIMAX_THRESHOLD = 80  # Passer à MiniMax quand empties < ce seuil

    def __init__(self, piece_type: str, name: str = "AmperesHybrid") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict[Tuple[str, str, Tuple[str, ...]], float] = OrderedDict()
        self._eval_cache: OrderedDict[Tuple[str, Tuple[str, ...]], float] = OrderedDict()
        self._tt: Dict[Tuple[Tuple[str, ...], str, int, str], float] = {}
        self._last_voltages: Dict[str, np.ndarray] = {}
        self._voltage_cache: Dict[Tuple, np.ndarray] = {}  # (target_piece, board_key) -> voltages

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
        self._tt.clear()
        self._voltage_cache.clear()

        if not empties:
            return self._to_action(0, n)

        # --- Coups gagnants immédiats ---
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(self._ordered_moves(board, my_wins, my_piece, geom)[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(self._ordered_moves(board, opp_wins, my_piece, geom)[0], n)

        # --- Filtrer les intermédiaires de bridges adverses intacts ---
        opp_bridge_cells = self._enemy_bridge_intermediates(board, opp_piece, geom)
        playable = [e for e in empties if e not in opp_bridge_cells] if opp_bridge_cells else empties
        if not playable:
            playable = empties

        # --- Choix de l'algorithme ---
        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        if len(playable) >= self.MINIMAX_THRESHOLD:
            return self._mcts_search(board, playable, my_piece, geom, deadline, n)
        else:
            return self._minimax_search(board, playable, my_piece, geom, deadline, n)

    # -----------------------------------------------------------------------
    # MCTS (early/mid game)
    # -----------------------------------------------------------------------

    def _mcts_search(self, board, empties, my_piece, geom, deadline, n):
        # Ordonnancement voltage seulement au root
        candidates = self._ordered_moves(board, empties, my_piece, geom)

        root = MCTSNode(
            move=None,
            parent=None,
            untried=candidates,
            to_play=my_piece,
        )

        empties_set = set(empties)

        while time.time() < deadline:
            node = root
            sim_board = list(board)
            sim_empties = set(empties_set)

            # 1. SÉLECTION
            while node.children:
                # Progressive widening : expand un nouvel enfant si sqrt(visits) > nb enfants
                if node.untried and len(node.children) < max(2, int(math.sqrt(node.visits + 1))):
                    break
                node = node.best_child(self.EXPLORATION)
                sim_board[node.move] = node.parent.to_play
                sim_empties.discard(node.move)

            # 2. EXPANSION
            if node.untried:
                mv = node.untried.pop(0)
                sim_board[mv] = node.to_play
                sim_empties.discard(mv)
                next_play = _other(node.to_play)

                # Ordonnancement léger pour les enfants (pas de voltage)
                child_candidates = self._light_ordered_moves(
                    sim_board, sim_empties, next_play, geom
                )

                child = MCTSNode(
                    move=mv,
                    parent=node,
                    untried=child_candidates,
                    to_play=next_play,
                )
                node.children.append(child)
                node = child

            # 3. SIMULATION — rollout léger (pas de voltage)
            result = self._rollout(sim_board, sim_empties, node.to_play, my_piece, geom)

            # 4. RÉTROPROPAGATION
            while node is not None:
                node.visits += 1
                node.value += result
                node = node.parent

        if root.children:
            best_move = root.most_visited_child().move
        else:
            best_move = candidates[0]

        return self._to_action(best_move, n)

    def _light_ordered_moves(self, board, empties_set, piece, geom):
        """Ordonnancement léger : ordre aléatoire (pas de biais)."""
        moves = list(empties_set)
        np.random.shuffle(moves)
        return moves

    def _rollout(self, board, empties_set, to_play, root_piece, geom):
        sim_board = list(board)
        sim_empties = list(empties_set)
        current = to_play

        if _has_won(sim_board, root_piece, geom):
            return 1.0
        if _has_won(sim_board, _other(root_piece), geom):
            return 0.0

        for _ in range(self.ROLLOUT_DEPTH):
            if not sim_empties:
                break

            # Coup aléatoire
            mv = sim_empties[np.random.randint(len(sim_empties))]

            sim_board[mv] = current
            sim_empties.remove(mv)

            # Vérifier si ce coup gagne
            if _has_won(sim_board, current, geom):
                return 1.0 if current == root_piece else 0.0

            current = _other(current)

        # Évaluation finale avec le modèle électrique (une seule fois)
        my_amp = self._calculate_amperage(tuple(sim_board), root_piece, geom)
        opp_amp = self._calculate_amperage(tuple(sim_board), _other(root_piece), geom)
        diff = my_amp - opp_amp
        return 1.0 / (1.0 + math.exp(-0.05 * diff))

    # -----------------------------------------------------------------------
    # MiniMax iterative deepening (late game)
    # -----------------------------------------------------------------------

    def _minimax_search(self, board, empties, my_piece, geom, deadline, n):
        opp_piece = _other(my_piece)
        ordered = self._ordered_moves(board, empties, my_piece, geom)

        best_move = ordered[0]
        best_score = float("-inf")

        for depth in range(1, 10):
            alpha = float("-inf")
            beta = float("inf")
            current_best_move = ordered[0]
            current_best_score = float("-inf")
            completed = True

            for mv in ordered:
                if time.time() > deadline:
                    completed = False
                    break

                board[mv] = my_piece
                if _has_won(board, my_piece, geom):
                    score = self.WIN_SCORE
                else:
                    rest = [e for e in empties if e != mv]
                    score = self._alpha_beta(
                        board, rest, opp_piece, depth - 1, alpha, beta,
                        my_piece, geom, deadline,
                    )
                board[mv] = "."

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = mv
                alpha = max(alpha, current_best_score)

            if completed or current_best_score > best_score:
                best_score = current_best_score
                best_move = current_best_move

            if time.time() > deadline:
                break

        return self._to_action(best_move, n)

    def _alpha_beta(self, board, empties, to_play, depth, alpha, beta, root_piece, geom, deadline):
        opp_root = _other(root_piece)
        if _has_won(board, root_piece, geom):
            return self.WIN_SCORE + depth
        if _has_won(board, opp_root, geom):
            return -self.WIN_SCORE - depth
        if depth == 0 or not empties:
            return self._evaluate_board(board, root_piece, geom)

        board_key = tuple(board)
        tt_key = (board_key, to_play, depth, root_piece)
        cached = self._tt.get(tt_key)
        if cached is not None:
            return cached

        maximizing = to_play == root_piece
        next_piece = _other(to_play)
        ordered = self._ordered_moves(board, empties, to_play, geom)

        if maximizing:
            best_score = float("-inf")
            for mv in ordered:
                if time.time() > deadline:
                    break
                board[mv] = to_play
                if _has_won(board, to_play, geom):
                    score = self.WIN_SCORE + depth
                else:
                    rest = [e for e in empties if e != mv]
                    score = self._alpha_beta(
                        board, rest, next_piece, depth - 1, alpha, beta,
                        root_piece, geom, deadline,
                    )
                board[mv] = "."
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            best_score = float("inf")
            for mv in ordered:
                if time.time() > deadline:
                    break
                board[mv] = to_play
                if _has_won(board, to_play, geom):
                    score = -self.WIN_SCORE - depth
                else:
                    rest = [e for e in empties if e != mv]
                    score = self._alpha_beta(
                        board, rest, next_piece, depth - 1, alpha, beta,
                        root_piece, geom, deadline,
                    )
                board[mv] = "."
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if alpha >= beta:
                    break

        self._tt[tt_key] = best_score
        return best_score

    # -----------------------------------------------------------------------
    # Évaluation
    # -----------------------------------------------------------------------

    def _evaluate_board(self, board, root_piece, geom):
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

    # -----------------------------------------------------------------------
    # Modèle électrique V3 + COO sparse + x0 + bridges alliés (symétriques)
    # -----------------------------------------------------------------------

    def _calculate_amperage(self, board_key, target_piece, geom):
        direction = "HAUTBAS" if target_piece == "R" else "GAUCHEDROITE"

        cache_key = (target_piece, direction, board_key)
        cached = self._amperage_cache.get(cache_key)
        if cached is not None:
            self._amperage_cache.move_to_end(cache_key)
            return cached

        n = geom["n"]
        size = geom["size"]
        opponent_piece = _other(target_piece)

        res_map = {target_piece: 0.001, ".": 1.0, opponent_piece: float("inf")}
        resistances = [res_map.get(p, float("inf")) for p in board_key]

        row_idx = []
        col_idx = []
        data_vals = []
        current = np.zeros(size)
        diag_sums = [0.0] * size
        diag_extra = [0.0] * size

        for idx in range(size):
            r_i = resistances[idx]
            if r_i == float("inf"):
                row_idx.append(idx)
                col_idx.append(idx)
                data_vals.append(1.0)
                continue

            for nb in geom["neighbors"][idx]:
                r_j = resistances[nb]
                if r_j == float("inf"):
                    continue
                conductance = 2.0 / (r_i + r_j)
                row_idx.append(idx)
                col_idx.append(nb)
                data_vals.append(-conductance)
                diag_sums[idx] += conductance

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

            diag_sums[idx] += source_c + sink_c
            current[idx] = source_c

        # Bridges alliés symétriques
        seen_pairs = set()
        for idx in range(size):
            if board_key[idx] != target_piece:
                continue
            for partner, c1, c2 in geom["bridge_links"][idx]:
                if board_key[partner] != target_piece:
                    continue
                if board_key[c1] == opponent_piece or board_key[c2] == opponent_piece:
                    continue
                pair = (min(idx, partner), max(idx, partner))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                v1, v2 = board_key[c1], board_key[c2]
                if v1 == "." and v2 == ".":
                    bridge_c = 400.0
                elif v1 == target_piece or v2 == target_piece:
                    bridge_c = 800.0
                else:
                    bridge_c = 100.0

                row_idx.extend([idx, partner])
                col_idx.extend([partner, idx])
                data_vals.extend([-bridge_c, -bridge_c])
                diag_extra[idx] += bridge_c
                diag_extra[partner] += bridge_c

        for idx in range(size):
            if resistances[idx] != float("inf"):
                row_idx.append(idx)
                col_idx.append(idx)
                data_vals.append(diag_sums[idx] + diag_extra[idx])

        g_csr = coo_matrix((data_vals, (row_idx, col_idx)), shape=(size, size)).tocsr()

        voltage_key = direction
        x0 = self._last_voltages.get(voltage_key)
        if x0 is None or len(x0) != size:
            x0 = geom["grad_row"] if direction == "HAUTBAS" else geom["grad_col"]

        try:
            voltages, _ = cg(g_csr, current, x0=x0, rtol=1e-5, maxiter=self.CG_MAXITER)
            self._last_voltages[voltage_key] = voltages
            self._voltage_cache[(target_piece, board_key)] = voltages

            total_current = 0.0
            for i in range(n):
                row = 0 if direction == "HAUTBAS" else i
                col = i if direction == "HAUTBAS" else 0
                idx = row * n + col
                if resistances[idx] != float("inf"):
                    total_current += (2.0 / resistances[idx]) * (1.0 - voltages[idx])
            result = float(total_current)
        except Exception:
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    # -----------------------------------------------------------------------
    # Ordonnancement V3 original
    # -----------------------------------------------------------------------

    def _ordered_moves(self, board, moves, piece, geom):
        opp = _other(piece)
        rescue_moves = self._bridge_responses(board, piece, geom)
        neighbors = geom["neighbors"]

        board_key = tuple(board)

        # Voltages alliés
        my_voltages = self._voltage_cache.get((piece, board_key))
        if my_voltages is None:
            self._calculate_amperage(board_key, piece, geom)
            my_voltages = self._voltage_cache.get((piece, board_key))

        scored = []

        for idx in moves:
            # Construire : gradient fort dans notre réseau à voltage ~0.5
            build = 0.0
            if my_voltages is not None:
                v = max(0.0, min(1.0, float(my_voltages[idx])))
                grad = 0.0
                for nb in neighbors[idx]:
                    if board[nb] == opp:
                        continue
                    nv = max(0.0, min(1.0, float(my_voltages[nb])))
                    grad = max(grad, abs(v - nv))
                centrality = 1.0 - abs(2.0 * v - 1.0)
                build = grad * centrality

            # Rescue reste prioritaire
            rescue = 18.0 if idx in rescue_moves else 0.0

            score = 4.0 * build + rescue
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _bridge_responses(self, board, piece, geom):
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

    # -----------------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------------

    def _winning_moves(self, board, empties, piece, geom):
        wins = []
        for idx in empties:
            board[idx] = piece
            if _has_won(board, piece, geom):
                wins.append(idx)
            board[idx] = "."
        return wins

    def _enemy_bridge_intermediates(self, board, opp_piece, geom):
        """Cases vides qui sont intermédiaires d'un bridge adverse intact."""
        blocked: Set[int] = set()
        for idx in range(len(board)):
            if board[idx] != opp_piece:
                continue
            for partner, c1, c2 in geom["bridge_links"][idx]:
                if board[partner] != opp_piece:
                    continue
                if board[c1] == "." and board[c2] == ".":
                    blocked.add(c1)
                    blocked.add(c2)
        return blocked

    def _extract_board(self, state, n):
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [idx for idx, cell in enumerate(board) if cell == "."]
        return board, empties

    def _to_action(self, idx, n):
        return StatelessAction({"piece": self.piece_type, "position": (idx // n, idx % n)})