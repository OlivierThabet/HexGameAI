'''
Refonte complète de ordered moves tout est dans la grille électrique
'''

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
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
    """Retourne (idx, partner, c1, c2) pour chaque bridge allié actif."""
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
    TIME_FRACTION = 0.1
    MIN_TIME_PER_MOVE = 0.5

    def __init__(self, piece_type: str, name: str = "AmperesBotV5") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict[Tuple[str, str, Tuple[str, ...]], float] = OrderedDict()
        self._eval_cache: OrderedDict[Tuple[str, Tuple[str, ...]], float] = OrderedDict()
        self._tt: Dict[Tuple[Tuple[str, ...], str, int, str], float] = {}
        self._last_voltages: Dict[str, np.ndarray] = {}

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

        # --- ACTIVE AREA PRUNING ---
        # Radically reduce the branching factor by filtering irrelevant empty spaces
        active_empties = self._get_active_area(board, empties, geom, padding=3)

        # 1-ply winning move checks (using the full empties list just to be safe)
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(self._ordered_moves(board, my_wins, my_piece, geom)[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(self._ordered_moves(board, opp_wins, my_piece, geom)[0], n)

        # --- Iterative deepening ---
        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        # FEED IT THE PRUNED LIST
        ordered = self._ordered_moves(board, active_empties, my_piece, geom)
        
        if not ordered: # Fallback if something weird happens and the box is empty
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

        if maximizing:
            best_score = float("-inf")
            for mv in self._ordered_moves(board, empties, to_play, geom):
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
            for mv in self._ordered_moves(board, empties, to_play, geom):
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

    def _get_active_area(self, board: List[str], empties: List[int], geom: dict, padding: int = 3) -> List[int]:
        """Filters empty cells to only those within a padded bounding box around existing pieces."""
        n = geom["n"]
        rows = geom["rows"]
        cols = geom["cols"]

        # Find all occupied positions (both your pieces and the opponent's)
        occupied = [i for i, cell in enumerate(board) if cell != "."]

        # If the board is completely empty (Turn 1), return the center area
        if not occupied:
            center = n // 2
            return [idx for idx in empties if abs(rows[idx] - center) <= padding and abs(cols[idx] - center) <= padding]

        # Find the mathematical bounding box of all played pieces
        min_r = min(rows[i] for i in occupied)
        max_r = max(rows[i] for i in occupied)
        min_c = min(cols[i] for i in occupied)
        max_c = max(cols[i] for i in occupied)

        # Expand the box by the padding and clamp it to the board edges
        min_r = max(0, min_r - padding)
        max_r = min(n - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(n - 1, max_c + padding)

        # Filter the empties list to only include cells inside this box
        active_empties = [
            idx for idx in empties
            if min_r <= rows[idx] <= max_r and min_c <= cols[idx] <= max_c
        ]

        return active_empties
    
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

    # -----------------------------------------------------------------------
    # Modèle électrique : V3 (résistances, moyenne harmonique) + matrice
    # creuse + x0 + bridges alliés comme connexions virtuelles
    # -----------------------------------------------------------------------

    def _calculate_amperage(self, board_key: Tuple[str, ...], target_piece: str, geom: dict) -> float:
        direction = "HAUTBAS" if target_piece == "R" else "GAUCHEDROITE"

        cache_key = (target_piece, direction, board_key)
        cached = self._amperage_cache.get(cache_key)
        if cached is not None:
            self._amperage_cache.move_to_end(cache_key)
            return cached

        try:
            n = geom["n"]
            size = geom["size"]
            opponent_piece = "B" if target_piece == "R" else "R"

            # 1. Bulletproof Resistance Mapping (defaults anything weird to inf)
            res_map = {target_piece: 0.001, ".": 1.0, opponent_piece: float("inf")}
            resistances = [res_map.get(p, float("inf")) for p in board_key]

            # Opponent bridges logic: Cut off paths if opponent has an intact bridge
            board_list = list(board_key)
            for idx in range(size):
                if board_key[idx] != opponent_piece:
                    continue
                for partner, c1, c2 in geom["bridge_links"][idx]:
                    if board_key[partner] == opponent_piece and board_key[c1] == "." and board_key[c2] == ".":
                        resistances[c1] = float("inf")
                        resistances[c2] = float("inf")

            # 2. Fast COO Matrix Assembly using Python Lists
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

            # 3. Virtual Connections (Allied Bridges)
            seen_pairs = set()
            for idx in range(size):
                if board_key[idx] != target_piece:
                    continue
                for partner, c1, c2 in geom["bridge_links"][idx]:
                    if board_key[partner] != target_piece:
                        continue
                    if board_key[c1] != opponent_piece and board_key[c2] != opponent_piece:
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

            # 4. Finalize Diagonals and Convert to CSR
            for idx in range(size):
                if resistances[idx] != float("inf"):
                    row_idx.append(idx)
                    col_idx.append(idx)
                    data_vals.append(diag_sums[idx] + diag_extra[idx])

            g_csr = coo_matrix((data_vals, (row_idx, col_idx)), shape=(size, size)).tocsr()

            # 5. Solve the System
            voltage_key = direction
            x0 = self._last_voltages.get(voltage_key)
            if x0 is None or len(x0) != size:
                x0 = geom["grad_row"] if direction == "HAUTBAS" else geom["grad_col"]

            voltages, _ = cg(g_csr, current, x0=x0, rtol=1e-5, maxiter=self.CG_MAXITER)
            self._last_voltages[voltage_key] = voltages

            total_current = 0.0
            for i in range(n):
                row = 0 if direction == "HAUTBAS" else i
                col = i if direction == "HAUTBAS" else 0
                idx = row * n + col
                r_i = resistances[idx]
                if r_i != float("inf"):
                    total_current += (2.0 / r_i) * (1.0 - voltages[idx])
            result = float(total_current)
            
        except Exception as e:
            # If anything fails (matrix math, weird board states), return 0.0 to prevent a fatal crash
            print(f"Amperage calculation failed: {e}")
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    # -----------------------------------------------------------------------
    # Ordonnancement V3 original
    # -----------------------------------------------------------------------

    def _ordered_moves(self, board: List[str], moves: List[int], piece: str, geom: dict) -> List[int]:
        opp = "B" if piece == "R" else "R"
        scored = []
        center = geom["center"]

        for idx in moves:
            own_n = 0
            opp_n = 0
            bridge_span = 0
            
            # 1. Check direct adjacency
            for nb in geom["neighbors"][idx]:
                cell = board[nb]
                if cell == piece:
                    own_n += 1
                elif cell == opp:
                    opp_n += 1

            # 2. Check for virtual connection spacing (taking space)
            # We don't check if the bridge is blocked here to save time, 
            # we just reward the optimal spacing geometry.
            for partner, _, _ in geom["bridge_links"][idx]:
                if board[partner] == piece:
                    bridge_span += 1

            row = geom["rows"][idx]
            col = geom["cols"][idx]
            
            # Forward progress (lane stay)
            progress = -abs(col - center) if piece == "R" else -abs(row - center)

            # --- THE NEW SCORING LOGIC ---
            # Penalize clumping (-1.5 * own_n)
            # Massively reward spanning and taking space (4.0 * bridge_span)
            # Still reward engaging the opponent to block them (2.0 * opp_n)
            score = (-1.5 * own_n) + (4.0 * bridge_span) + (2.0 * opp_n) + (0.30 * progress)
            
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _winning_moves(self, board: List[str], empties: List[int], piece: str, geom: dict) -> List[int]:
        wins = []
        for idx in empties:
            board[idx] = piece
            if _has_won(board, piece, geom):
                wins.append(idx)
            board[idx] = "."
        return wins

    def _extract_board(self, state: GameStateHex, n: int) -> Tuple[List[str], List[int]]:
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [idx for idx, cell in enumerate(board) if cell == "."]
        return board, empties

    def _to_action(self, idx: int, n: int) -> StatelessAction:
        return StatelessAction({"piece": self.piece_type, "position": (idx // n, idx % n)})