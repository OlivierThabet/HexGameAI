'''
Agent hybride MCTS → MiniMax.
- Phase 1 (First 30 moves): Heavy Leaf MCTS
- Phase 2 (Late game): Alpha-Beta Minimax
- Bridges + edge templates II/III/IV + double bridges in amperage model
- Evaluation : Pure Amperage (Variable Resistivity + spsolve)
- Ordering: Lean pure voltage (no intrusion penalties)
- Forced Moves: Wins and Blocks only (Bridge rescue disabled)
'''

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

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


def _cell(r: int, c: int, n: int) -> int:
    if 0 <= r < n and 0 <= c < n:
        return r * n + c
    return -1


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
                if not (0 <= pr < n and 0 <= pc < n and
                        0 <= c1r < n and 0 <= c1c < n and
                        0 <= c2r < n and 0 <= c2c < n):
                    continue
                key = (pr * n + pc, c1r * n + c1c, c2r * n + c2c)
                if key not in seen:
                    seen.add(key)
                    bridge_links[idx].append(key)
    grad_row = np.array([1.0 - r / (n - 1) for r in range(n) for _ in range(n)])
    grad_col = np.array([1.0 - c / (n - 1) for _ in range(n) for c in range(n)])
    geom = {
        "n": n, "size": size, "neighbors": neighbors,
        "bridge_links": bridge_links, "rows": rows, "cols": cols,
        "center": (n - 1) / 2.0, "grad_row": grad_row, "grad_col": grad_col,
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

# ===========================================================================
# MCTS Node
# ===========================================================================

class MCTSNode:
    def __init__(self, move: int = None, parent: 'MCTSNode' = None, state_board: List[str] = None, to_play: str = None):
        self.move = move
        self.parent = parent
        self.board = state_board  
        self.to_play = to_play    
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0          
        self.untried_moves: List[int] = [] 
        self.is_terminal = False
        self.terminal_value = 0.0 

    def ucb1(self, exploration_c: float = 0.5, is_minimizing: bool = False) -> float:
        if self.visits == 0:
            return float('inf')
        
        win_rate = self.value / self.visits
        exploit = (1.0 - win_rate) if is_minimizing else win_rate
        explore = exploration_c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

# ===========================================================================
# Edge template geometry (used by amperage model only)
# ===========================================================================

def _get_template_IV_instances(n: int, piece: str):
    """Template IV-1-a (defendable): stone on row 3, carrier on rows 0-2."""
    insts = []
    def _try(stone, cells, k0, k1):
        if stone >= 0 and k0 >= 0 and k1 >= 0 and all(x >= 0 for x in cells):
            insts.append((stone, cells, k0, k1))

    if piece == "R":
        for c in range(n):
            s = _cell(3, c, n)
            k0 = _cell(2, c, n)
            k1 = _cell(2, c + 1, n)
            rest = [_cell(1, c, n), _cell(1, c+1, n), _cell(1, c+2, n),
                    _cell(0, c, n), _cell(0, c+1, n), _cell(0, c+2, n), _cell(0, c+3, n)]
            _try(s, [k0, k1] + rest, k0, k1)
            k0m = _cell(2, c - 1, n)
            k1m = _cell(2, c, n)
            restm = [_cell(1, c-2, n), _cell(1, c-1, n), _cell(1, c, n),
                     _cell(0, c-3, n), _cell(0, c-2, n), _cell(0, c-1, n), _cell(0, c, n)]
            _try(s, [k0m, k1m] + restm, k0m, k1m)
        for c in range(n):
            s = _cell(n - 4, c, n)
            k0 = _cell(n-3, c-1, n)
            k1 = _cell(n-3, c, n)
            rest = [_cell(n-2, c-2, n), _cell(n-2, c-1, n), _cell(n-2, c, n),
                    _cell(n-1, c-3, n), _cell(n-1, c-2, n), _cell(n-1, c-1, n), _cell(n-1, c, n)]
            _try(s, [k0, k1] + rest, k0, k1)
            k0m = _cell(n-3, c, n)
            k1m = _cell(n-3, c+1, n)
            restm = [_cell(n-2, c, n), _cell(n-2, c+1, n), _cell(n-2, c+2, n),
                     _cell(n-1, c, n), _cell(n-1, c+1, n), _cell(n-1, c+2, n), _cell(n-1, c+3, n)]
            _try(s, [k0m, k1m] + restm, k0m, k1m)
    else:
        for r in range(n):
            s = _cell(r, 3, n)
            k0 = _cell(r, 2, n)
            k1 = _cell(r + 1, 2, n)
            rest = [_cell(r, 1, n), _cell(r+1, 1, n), _cell(r+2, 1, n),
                    _cell(r, 0, n), _cell(r+1, 0, n), _cell(r+2, 0, n), _cell(r+3, 0, n)]
            _try(s, [k0, k1] + rest, k0, k1)
            k0m = _cell(r - 1, 2, n)
            k1m = _cell(r, 2, n)
            restm = [_cell(r-2, 1, n), _cell(r-1, 1, n), _cell(r, 1, n),
                     _cell(r-3, 0, n), _cell(r-2, 0, n), _cell(r-1, 0, n), _cell(r, 0, n)]
            _try(s, [k0m, k1m] + restm, k0m, k1m)
        for r in range(n):
            s = _cell(r, n - 4, n)
            k0 = _cell(r - 1, n-3, n)
            k1 = _cell(r, n-3, n)
            rest = [_cell(r-2, n-2, n), _cell(r-1, n-2, n), _cell(r, n-2, n),
                    _cell(r-3, n-1, n), _cell(r-2, n-1, n), _cell(r-1, n-1, n), _cell(r, n-1, n)]
            _try(s, [k0, k1] + rest, k0, k1)
            k0m = _cell(r, n-3, n)
            k1m = _cell(r + 1, n-3, n)
            restm = [_cell(r, n-2, n), _cell(r+1, n-2, n), _cell(r+2, n-2, n),
                     _cell(r, n-1, n), _cell(r+1, n-1, n), _cell(r+2, n-1, n), _cell(r+3, n-1, n)]
            _try(s, [k0m, k1m] + restm, k0m, k1m)
    return insts


# ===========================================================================
# Player
# ===========================================================================

class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    MAX_EVAL_CACHE = 50000
    TIME_FRACTION = 0.06
    MIN_TIME_PER_MOVE = 0.5

    def __init__(self, piece_type: str, name: str = "AmperesMCTS-MiniMax") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict = OrderedDict()
        self._eval_cache: OrderedDict = OrderedDict()
        self._tt: Dict = {}
        self._voltage_cache: Dict[Tuple, np.ndarray] = {}

    def compute_action(self, current_state: GameStateHex,
                       remaining_time: float = 15 * 60, **kwargs) -> Action:
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

        # --- Opening: center ---
        my_stones = sum(1 for cell in board if cell == my_piece)
        if my_stones == 0:
            center = n // 2
            center_idx = center * n + center
            if board[center_idx] == ".":
                return self._to_action(center_idx, n)

        # --- 🚨 FORCED MOVES TOOL 🚨 ---
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            print(f"\n[{self.name}] ⚡ FORCED MOVE: Immediate Win at ({my_wins[0] // n}, {my_wins[0] % n})")
            return self._to_action(my_wins[0], n)
            
        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            print(f"\n[{self.name}] 🛡️ FORCED MOVE: Blocking Opponent Win at ({opp_wins[0] // n}, {opp_wins[0] % n})")
            return self._to_action(opp_wins[0], n)

        # Un peu trash ca --- Bridge defense (only forced move besides wins) ---
        #rescue_moves = self._bridge_responses(board, my_piece, geom)
        #if rescue_moves:
        #    rescue_list = [m for m in rescue_moves if m in empties]
        #    if rescue_list:
        #        mv = self._pick_best(board, rescue_list, my_piece, geom)
        #        print(f"\n[{self.name}] 🌉 FORCED MOVE: Bridge Rescue at ({mv // n}, {mv % n})")
        #        return self._to_action(mv, n)

        # --- Algorithm choice: MCTS -> MiniMax ---
        total_cells = geom["size"]
        moves_played = total_cells - len(empties)

        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        if moves_played < 30:
            # PHASE 1: Heavy Leaf MCTS for the wide open early/mid game
            chosen = self._mcts_search(board, empties, my_piece, geom, deadline)
        else:
            # PHASE 2: Minimax Alpha-Beta for tactical late game
            print(f"\n[{self.name}] Switching to Minimax (Moves Played: {moves_played})")
            chosen = self._minimax_pick(board, empties, my_piece, geom, deadline)

        # --- Post-search override: critical template blocking ---
        override = self._critical_template_override(
            board, chosen, empties, my_piece, opp_piece, geom)
        if override is not None:
            chosen = override

        return self._to_action(chosen, n)

    # -----------------------------------------------------------------------
    # MCTS (early/mid game)
    # -----------------------------------------------------------------------

    def _mcts_search(self, board, empties, root_piece, geom, deadline):
        root = MCTSNode(state_board=list(board), to_play=root_piece)
        root.untried_moves = self._ordered_moves(board, empties, root_piece, geom)
        
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            node = root
            
            # 1. SELECTION
            while not node.untried_moves and node.children and not node.is_terminal:
                is_minimizing = (node.to_play != root_piece)
                node = max(node.children, key=lambda c: c.ucb1(exploration_c=0.5, is_minimizing=is_minimizing))

            # 2. EXPANSION
            if node.untried_moves and not node.is_terminal:
                move = node.untried_moves.pop(0) 
                next_state = list(node.board)
                next_state[move] = node.to_play
                next_piece = _other(node.to_play)
                
                child = MCTSNode(move=move, parent=node, state_board=next_state, to_play=next_piece)
                
                if _has_won(next_state, node.to_play, geom):
                    child.is_terminal = True
                    child.terminal_value = 1.0 if node.to_play == root_piece else 0.0
                else:
                    child.untried_moves = [m for m in empties if next_state[m] == "."]
                
                node.children.append(child)
                node = child

            # 3. HEAVY LEAF EVALUATION
            if node.is_terminal:
                v = node.terminal_value
            else:
                amp_diff = self._evaluate_board(node.board, root_piece, geom)
                # Temperature 50.0 to widen probabilities
                v = 1.0 / (1.0 + math.exp(-50.0 * amp_diff)) 

            # 4. BACKPROPAGATION
            while node is not None:
                node.visits += 1
                node.value += v 
                node = node.parent

        self._debug_mcts_tree(root, iterations, geom["n"])

        if not root.children:
            return root.untried_moves[0] if root.untried_moves else 0
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _debug_mcts_tree(self, root: MCTSNode, iterations: int, n: int):
        print(f"\n[{self.name}] MCTS Complete: {iterations} iterations.")
        if not root.children: return

        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        print(f"{'Move':>8} | {'Visits':>8} | {'P(Win)':>8} | {'UCB1':>8}")
        print("-" * 40)
        for child in sorted_children[:5]: 
            move_str = f"({child.move // n}, {child.move % n})"
            win_rate = child.value / child.visits if child.visits > 0 else 0.0
            ucb = child.ucb1() if child.visits > 0 else 0.0
            print(f"{move_str:>8} | {child.visits:>8} | {win_rate:>7.2%} | {ucb:>8.3f}")

    # -----------------------------------------------------------------------
    # Minimax (late game)
    # -----------------------------------------------------------------------

    def _minimax_pick(self, board, empties, my_piece, geom, deadline):
        opp_piece = _other(my_piece)
        ordered = self._ordered_moves(board, empties, my_piece, geom)
        best_move = ordered[0]
        best_score = float("-inf")
        for depth in range(1, 10):
            alpha, beta = float("-inf"), float("inf")
            cur_best_move, cur_best_score = ordered[0], float("-inf")
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
                    score = self._alpha_beta(board, rest, opp_piece, depth - 1,
                                             alpha, beta, my_piece, geom, deadline)
                board[mv] = "."
                if score > cur_best_score:
                    cur_best_score = score
                    cur_best_move = mv
                alpha = max(alpha, cur_best_score)
            if completed or cur_best_score > best_score:
                best_score = cur_best_score
                best_move = cur_best_move
            if time.time() > deadline:
                break
        return best_move

    def _alpha_beta(self, board, empties, to_play, depth, alpha, beta,
                    root_piece, geom, deadline):
        opp_root = _other(root_piece)
        if _has_won(board, root_piece, geom):
            return self.WIN_SCORE + depth
        if _has_won(board, opp_root, geom):
            return -self.WIN_SCORE - depth
        if depth == 0 or not empties:
            return self._evaluate_board(board, root_piece, geom)

        tt_key = (tuple(board), to_play, depth, root_piece)
        cached = self._tt.get(tt_key)
        if cached is not None:
            return cached

        maximizing = to_play == root_piece
        next_piece = _other(to_play)
        ordered = self._ordered_moves(board, empties, to_play, geom)

        if maximizing:
            best = float("-inf")
            for mv in ordered:
                if time.time() > deadline:
                    break
                board[mv] = to_play
                sc = (self.WIN_SCORE + depth) if _has_won(board, to_play, geom) else \
                    self._alpha_beta(board, [e for e in empties if e != mv],
                                     next_piece, depth - 1, alpha, beta,
                                     root_piece, geom, deadline)
                board[mv] = "."
                if sc > best:
                    best = sc
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = float("inf")
            for mv in ordered:
                if time.time() > deadline:
                    break
                board[mv] = to_play
                sc = (-self.WIN_SCORE - depth) if _has_won(board, to_play, geom) else \
                    self._alpha_beta(board, [e for e in empties if e != mv],
                                     next_piece, depth - 1, alpha, beta,
                                     root_piece, geom, deadline)
                board[mv] = "."
                if sc < best:
                    best = sc
                beta = min(beta, best)
                if alpha >= beta:
                    break

        self._tt[tt_key] = best
        return best

    # -----------------------------------------------------------------------
    # Evaluation (Pure Amperage difference)
    # -----------------------------------------------------------------------

    def _evaluate_board(self, board, root_piece, geom):
        board_key = tuple(board)
        cache_key = (root_piece, board_key)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            self._eval_cache.move_to_end(cache_key)
            return cached
            
        opp = _other(root_piece)
        score = self._calculate_amperage(board_key, root_piece, geom) - \
                self._calculate_amperage(board_key, opp, geom)
                     
        self._eval_cache[cache_key] = score
        if len(self._eval_cache) > self.MAX_EVAL_CACHE:
            self._eval_cache.popitem(last=False)
        return score

    # -----------------------------------------------------------------------
    # Electrical model with Variable Resistivity + Exact Solver
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
        opp = _other(target_piece)
        center = geom["center"]
        
        resistances = []
        for idx, p in enumerate(board_key):
            if p == target_piece:
                resistances.append(0.001)
            elif p == opp:
                resistances.append(float("inf"))
            else:
                row, col = geom["rows"][idx], geom["cols"][idx]
                row_dist = abs(row - center) / center
                col_dist = abs(col - center) / center
                dist_sq = (row_dist**2 + col_dist**2) / 2.0
                r_val = 1.0 + 5.0 * dist_sq
                
                opp_adj = sum(1 for nb in geom["neighbors"][idx] if board_key[nb] == opp)
                r_val += 3.0 * opp_adj 
                
                resistances.append(r_val)

        row_idx, col_idx, data_vals = [], [], []
        current = np.zeros(size)
        diag_sums = [0.0] * size
        diag_extra = [0.0] * size

        R_CONTACT = 10.0 

        for idx in range(size):
            r_i = resistances[idx]
            if r_i == float("inf"):
                row_idx.append(idx); col_idx.append(idx); data_vals.append(1.0)
                continue
            for nb in geom["neighbors"][idx]:
                r_j = resistances[nb]
                if r_j == float("inf"):
                    continue
                c_val = 2.0 / (r_i + r_j)
                row_idx.append(idx); col_idx.append(nb); data_vals.append(-c_val)
                diag_sums[idx] += c_val
                
            src, snk = 0.0, 0.0
            row, col = geom["rows"][idx], geom["cols"][idx]
            
            if direction == "HAUTBAS":
                if row == 0: src = 2.0 / (R_CONTACT + r_i)
                if row == n - 1: snk = 2.0 / (R_CONTACT + r_i)
            else:
                if col == 0: src = 2.0 / (R_CONTACT + r_i)
                if col == n - 1: snk = 2.0 / (R_CONTACT + r_i)
                
            diag_sums[idx] += src + snk
            current[idx] = src 

        seen_pairs = set()
        for idx in range(size):
            if board_key[idx] != target_piece:
                continue
            for partner, c1, c2 in geom["bridge_links"][idx]:
                if board_key[partner] != target_piece:
                    continue
                if board_key[c1] == opp or board_key[c2] == opp:
                    continue
                pair = (min(idx, partner), max(idx, partner))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                v1, v2 = board_key[c1], board_key[c2]
                if v1 == target_piece or v2 == target_piece:
                    continue 
                
                bc = 400.0 if (v1 == "." and v2 == ".") else 100.0
                    
                row_idx.extend([idx, partner])
                col_idx.extend([partner, idx])
                data_vals.extend([-bc, -bc])
                diag_extra[idx] += bc
                diag_extra[partner] += bc

        for idx in range(size):
            if resistances[idx] != float("inf"):
                row_idx.append(idx); col_idx.append(idx)
                data_vals.append(diag_sums[idx] + diag_extra[idx])

        g_csr = coo_matrix((data_vals, (row_idx, col_idx)), shape=(size, size)).tocsr()

        try:
            voltages = spsolve(g_csr, current)
            self._voltage_cache[(target_piece, board_key)] = voltages
            
            total = 0.0
            for i in range(n):
                row = 0 if direction == "HAUTBAS" else i
                col = i if direction == "HAUTBAS" else 0
                idx = row * n + col
                if resistances[idx] != float("inf"):
                    total += (2.0 / (R_CONTACT + resistances[idx])) * (1.0 - voltages[idx])
            result = float(total)
        except Exception:
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    # -----------------------------------------------------------------------
    # Move ordering (Lean: pure voltage only)
    # -----------------------------------------------------------------------

    def _ordered_moves(self, board, moves, piece, geom):
        board_key = tuple(board)
        opp = _other(piece)

        my_v = self._voltage_cache.get((piece, board_key))
        if my_v is None:
            self._calculate_amperage(board_key, piece, geom)
            my_v = self._voltage_cache.get((piece, board_key))

        opp_v = self._voltage_cache.get((opp, board_key))
        if opp_v is None:
            self._calculate_amperage(board_key, opp, geom)
            opp_v = self._voltage_cache.get((opp, board_key))

        scored = []
        for idx in moves:
            my_score = 0.0
            opp_score = 0.0
            if my_v is not None:
                val = max(0.0, min(1.0, float(my_v[idx])))
                my_score = val * (1.0 - val)
            if opp_v is not None:
                val = max(0.0, min(1.0, float(opp_v[idx])))
                opp_score = val * (1.0 - val)
                
            scored.append((my_score + opp_score, idx))
            
        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    # -----------------------------------------------------------------------
    # Utilities & Template logic
    # -----------------------------------------------------------------------

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
        
    def _pick_best(self, board, move_list, piece, geom):
        if len(move_list) == 1:
            return move_list[0]
        best = move_list[0]
        best_sc = float("-inf")
        for mv in move_list:
            board[mv] = piece
            sc = self._evaluate_board(board, piece, geom)
            board[mv] = "."
            if sc > best_sc:
                best_sc = sc
                best = mv
        return best

    def _winning_moves(self, board, empties, piece, geom):
        wins = []
        for idx in empties:
            board[idx] = piece
            if _has_won(board, piece, geom):
                wins.append(idx)
            board[idx] = "."
        return wins

    def _extract_board(self, state, n):
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [idx for idx, cell in enumerate(board) if cell == "."]
        return board, empties

    def _to_action(self, idx, n):
        return StatelessAction({"piece": self.piece_type, "position": (idx // n, idx % n)})

    def _critical_template_override(self, board, chosen_move, empties, my_piece, opp_piece, geom):
        n = geom["n"]
        board_key = tuple(board)

        opp_v = self._voltage_cache.get((opp_piece, board_key))
        if opp_v is None:
            self._calculate_amperage(board_key, opp_piece, geom)
            opp_v = self._voltage_cache.get((opp_piece, board_key))
        if opp_v is None:
            return None

        VOLTAGE_LOW = 0.2
        VOLTAGE_HIGH = 0.8
        critical_targets: List[Tuple[float, int]] = []

        for stone, carrier, k0, k1 in _get_template_IV_instances(n, opp_piece):
            if board[stone] != opp_piece:
                continue
            if any(board[ci] == my_piece for ci in carrier):
                continue
            v = float(opp_v[stone])
            if v < VOLTAGE_LOW or v > VOLTAGE_HIGH:
                continue
            density = v * (1.0 - v)
            if board[k0] == ".":
                critical_targets.append((density, k0))
            if board[k1] == ".":
                critical_targets.append((density, k1))

        if not critical_targets:
            return None

        chosen_set = {chosen_move}
        for _, target in critical_targets:
            if target in chosen_set:
                return None 

        critical_targets.sort(reverse=True)
        candidates = []
        seen = set()
        for _, target in critical_targets:
            if target not in seen and target in empties:
                seen.add(target)
                candidates.append(target)

        if not candidates:
            return None

        board[chosen_move] = my_piece
        chosen_score = self._evaluate_board(board, my_piece, geom)
        board[chosen_move] = "."

        best_block = candidates[0]
        best_block_score = float("-inf")
        for mv in candidates[:4]: 
            board[mv] = my_piece
            sc = self._evaluate_board(board, my_piece, geom)
            board[mv] = "."
            if sc > best_block_score:
                best_block_score = sc
                best_block = mv

        if best_block_score >= chosen_score * 0.8 - 1.0:
            return best_block

        return None