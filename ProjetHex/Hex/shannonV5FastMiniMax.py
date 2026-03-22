'''
De V4, Ouverture étendue — Le seuil passe de 4 à 6 pierres jouées, avec un anneau de candidats plus large (5×5 autour du centre). Les premiers coups se jouent quasi instantanément sans alpha-bêta.
Top-K pruning — Seuls les 12 meilleurs coups (selon l'ordonnancement heuristique) sont explorés à chaque nœud. Sur un plateau de 121 cases, ça réduit le branching factor de ~120 à 12, soit ~100x moins de nœuds à profondeur 2.
Iterative deepening + gestion du temps — Au lieu d'une profondeur fixe, on commence à profondeur 1 et on augmente tant qu'il reste du temps (6% du temps restant par coup, minimum 0.5s). Ça garantit toujours une réponse rapide, et on va plus profond quand le temps le permet (milieu/fin de partie). Une history heuristic récompense les coups qui ont été bons aux profondeurs précédentes pour améliorer le pruning.
++++ AJOUT DE BRIDGES
'''

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import lil_matrix
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


def _virtual_connections(board: List[str], piece: str, geom: dict) -> List[Tuple[int, int, int, int]]:
    """Retourne (idx, partner, c1, c2) pour chaque bridge actif du joueur."""
    opp = _other(piece)
    connections: List[Tuple[int, int, int, int]] = []

    for idx in range(len(board)):
        if board[idx] != piece:
            continue
        for partner, c1, c2 in geom["bridge_links"][idx]:
            if board[partner] != piece:
                continue
            if board[c1] != opp and board[c2] != opp:
                connections.append((idx, partner, c1, c2))

    return connections


class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    MAX_EVAL_CACHE = 50000
    CG_MAXITER = 40
    TOP_K = 12
    TIME_FRACTION = 0.06
    MIN_TIME_PER_MOVE = 0.5

    def __init__(self, piece_type: str, name: str = "AmperesBotV4") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict[Tuple[str, str, Tuple[str, ...]], float] = OrderedDict()
        self._eval_cache: OrderedDict[Tuple[str, Tuple[str, ...]], float] = OrderedDict()
        self._tt: Dict[Tuple[Tuple[str, ...], str, int, str], float] = {}
        self._last_voltages: Dict[str, np.ndarray] = {}
        self._history: Dict[int, float] = {}

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

        if not empties:
            return self._to_action(0, n)

        # --- Ouverture étendue ---
        opening = self._opening_move(board, empties, my_piece, geom)
        if opening is not None:
            return self._to_action(opening, n)

        # --- Coups gagnants immédiats ---
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(self._ordered_moves(board, my_wins, my_piece, geom)[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(self._ordered_moves(board, opp_wins, my_piece, geom)[0], n)

        # --- Iterative deepening avec gestion du temps ---
        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        ordered = self._ordered_moves(board, empties, my_piece, geom)
        candidates = ordered[: self.TOP_K]

        best_move = candidates[0]
        best_score = float("-inf")

        for depth in range(1, 10):
            alpha = float("-inf")
            beta = float("inf")
            current_best_move = candidates[0]
            current_best_score = float("-inf")
            completed = True

            for mv in candidates:
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
                self._history[best_move] = self._history.get(best_move, 0.0) + depth * depth

            if time.time() > deadline:
                break

        return self._to_action(best_move, n)

    def _alpha_beta(
        self,
        board: List[str],
        empties: List[int],
        to_play: str,
        depth: int,
        alpha: float,
        beta: float,
        root_piece: str,
        geom: dict,
        deadline: float,
    ) -> float:
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
        ordered = self._ordered_moves(board, empties, to_play, geom)[: self.TOP_K]

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
        opponent_piece = _other(target_piece)
        board_list = list(board_key)

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

        g_mat = lil_matrix((size, size))
        current = np.zeros(size)
        diag_extra = np.zeros(size)

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
                g_mat[idx, nb] = -c
                diag_sum += c

            source_c = 0.0
            sink_c = 0.0
            row = geom["rows"][idx]
            col = geom["cols"][idx]
            if direction == "HAUTBAS":
                if row == 0:
                    source_c = source_conductance(cell_i)
                if row == n - 1:
                    sink_c = source_conductance(cell_i)
            else:
                if col == 0:
                    source_c = source_conductance(cell_i)
                if col == n - 1:
                    sink_c = source_conductance(cell_i)

            g_mat[idx, idx] = diag_sum + source_c + sink_c
            current[idx] = source_c

        # --- Arêtes virtuelles pour les bridges ---
        vconns = _virtual_connections(board_list, target_piece, geom)
        seen_pairs: Set[Tuple[int, int]] = set()

        for idx, partner, c1, c2 in vconns:
            pair = (min(idx, partner), max(idx, partner))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            v1 = board_key[c1]
            v2 = board_key[c2]

            # Deux intermédiaires libres → bridge parfait, quasi court-circuit
            if v1 == "." and v2 == ".":
                bridge_c = 400.0
            # Un intermédiaire allié → encore plus solide
            elif v1 == target_piece or v2 == target_piece:
                bridge_c = 800.0
            else:
                bridge_c = 100.0

            g_mat[idx, partner] = g_mat[idx, partner] - bridge_c
            g_mat[partner, idx] = g_mat[partner, idx] - bridge_c
            diag_extra[idx] += bridge_c
            diag_extra[partner] += bridge_c

        for idx in range(size):
            if diag_extra[idx] != 0.0 and board_key[idx] != opponent_piece:
                g_mat[idx, idx] = g_mat[idx, idx] + diag_extra[idx]

        g_csr = g_mat.tocsr()

        voltage_key = direction
        x0 = self._last_voltages.get(voltage_key)
        if x0 is None or x0.shape[0] != size:
            x0 = geom["grad_row"] if direction == "HAUTBAS" else geom["grad_col"]

        try:
            voltages, _ = cg(g_csr, current, x0=x0, rtol=1e-5, maxiter=self.CG_MAXITER)
            self._last_voltages[voltage_key] = voltages

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

            hist = self._history.get(idx, 0.0)
            score = (
                3.2 * own_n
                + 2.8 * opp_n
                + 0.25 * center_bias
                + 0.30 * progress
                + bridge_bonus
                + 0.1 * hist
            )
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
            if _has_won(board, piece, geom):
                wins.append(idx)
            board[idx] = "."
        return wins

    def _opening_move(self, board: List[str], empties: List[int], piece: str, geom: dict) -> Optional[int]:
        stones_played = len(board) - len(empties)
        if stones_played >= 6:
            return None

        n = geom["n"]
        center = n // 2
        candidates = set()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = center + dr, center + dc
                if 0 <= r < n and 0 <= c < n:
                    candidates.add(r * n + c)

        available = [idx for idx in candidates if board[idx] == "."]
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