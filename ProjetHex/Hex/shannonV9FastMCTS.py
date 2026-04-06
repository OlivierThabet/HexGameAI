"""
Équivalent MCTS du V5 MiniMax optimisé.
Partage la même évaluation (COO Matrix), le même ordonnancement (Spanning),
le même Active Area Pruning et la même gestion de temps dynamique.
"""

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

    def __init__(
        self,
        move: Optional[int],
        parent: Optional["MCTSNode"],
        untried: List[int],
        to_play: str,
    ) -> None:
        self.move = move
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        # Inversion pour que pop() prenne le meilleur coup en premier
        self.untried = untried[::-1] 
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
# Joueur MCTS
# ---------------------------------------------------------------------------

class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    CG_MAXITER = 40
    TIME_FRACTION = 0.1
    MIN_TIME_PER_MOVE = 0.5
    EXPLORATION = 1.414
    ROLLOUT_DEPTH = 6

    def __init__(self, piece_type: str, name: str = "AmperesMCTS_V5") -> None:
        super().__init__(piece_type, name)
        self._amperage_cache: OrderedDict[Tuple[str, str, Tuple[str, ...]], float] = OrderedDict()
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

        if not empties:
            return self._to_action(0, n)
        
        # --- ACTIVE AREA PRUNING ---
        active_empties = self._get_active_area(board, empties, geom, padding=3)
        
        # 1-ply winning move checks
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(my_wins[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(opp_wins[0], n)

        # --- Iterative deepening ---
        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        # Initialisation Racine : On prend la zone active et on mélange
        candidates = list(active_empties)
        if not candidates: 
            candidates = list(empties)
        np.random.shuffle(candidates)

        root = MCTSNode(
            move=None,
            parent=None,
            untried=candidates, # Plus besoin du [::-1] vu que c'est aléatoire
            to_play=my_piece,
        )

        # --- MCTS LOOP ---
        while time.time() < deadline:
            node = root
            sim_board = list(board)
            sim_empties = list(empties)

            # 1. SÉLECTION
            while not node.untried and node.children:
                node = node.best_child(self.EXPLORATION)
                sim_board[node.move] = node.parent.to_play
                sim_empties = [e for e in sim_empties if e != node.move]

            # 2. EXPANSION
            if node.untried:
                mv = node.untried.pop() # Retire le meilleur coup restant grâce au [::-1]
                sim_board[mv] = node.to_play
                sim_empties = [e for e in sim_empties if e != mv]
                next_play = _other(node.to_play)

                # Expansion purement aléatoire
                child_candidates = list(sim_empties)
                np.random.shuffle(child_candidates)

                child = MCTSNode(
                    move=mv,
                    parent=node,
                    untried=child_candidates,
                    to_play=next_play,
                )
                node.children.append(child)
                node = child

            # 3. SIMULATION
            result = self._rollout(sim_board, sim_empties, node.to_play, my_piece, geom)

            # 4. RÉTROPROPAGATION
            while node is not None:
                node.visits += 1
                node.value += result
                node = node.parent

        # Choisir le coup le plus visité
        if root.children:
            best_move = root.most_visited_child().move
        else:
            best_move = candidates[0]

        return self._to_action(best_move, n)

    # -----------------------------------------------------------------------
    # Rollout : coups semi-guidés puis évaluation heuristique
    # -----------------------------------------------------------------------

    def _rollout(
        self,
        board: List[str],
        empties: List[int],
        to_play: str,
        root_piece: str,
        geom: dict,
    ) -> float:
        sim_board = list(board)
        sim_empties = list(empties)
        current = to_play

        # Vérification initiale de l'état du plateau
        if _has_won(sim_board, root_piece, geom):
            return 1.0
        if _has_won(sim_board, _other(root_piece), geom):
            return 0.0

        for _ in range(self.ROLLOUT_DEPTH):
            if not sim_empties:
                break

            # 1. Choix aléatoire pur et ultra-rapide
            rand_idx = np.random.randint(len(sim_empties))
            mv = sim_empties.pop(rand_idx)

            sim_board[mv] = current
            
            # 2. Vérification de victoire UNIQUEMENT pour le coup qu'on vient de jouer
            if _has_won(sim_board, current, geom):
                return 1.0 if current == root_piece else 0.0

            current = _other(current)

        # 3. Évaluation heuristique avec la matrice rapide
        my_amp = self._calculate_amperage(tuple(sim_board), root_piece, geom)
        opp_amp = self._calculate_amperage(tuple(sim_board), _other(root_piece), geom)

        diff = my_amp - opp_amp
        return 1.0 / (1.0 + math.exp(-0.05 * diff))

    # -----------------------------------------------------------------------
    # Helpers : Active Area & Ordonnancement (V5)
    # -----------------------------------------------------------------------

    def _get_active_area(self, board: List[str], empties: List[int], geom: dict, padding: int = 3) -> List[int]:
        n = geom["n"]
        rows = geom["rows"]
        cols = geom["cols"]

        occupied = [i for i, cell in enumerate(board) if cell != "."]

        if not occupied:
            center = n // 2
            return [idx for idx in empties if abs(rows[idx] - center) <= padding and abs(cols[idx] - center) <= padding]

        min_r = min(rows[i] for i in occupied)
        max_r = max(rows[i] for i in occupied)
        min_c = min(cols[i] for i in occupied)
        max_c = max(cols[i] for i in occupied)

        min_r = max(0, min_r - padding)
        max_r = min(n - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(n - 1, max_c + padding)

        active_empties = [
            idx for idx in empties
            if min_r <= rows[idx] <= max_r and min_c <= cols[idx] <= max_c
        ]

        return active_empties

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

    # -----------------------------------------------------------------------
    # Modèle électrique V5 : COO Matrix rapide + Gestion des erreurs
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

            res_map = {target_piece: 0.001, ".": 1.0, opponent_piece: float("inf")}
            resistances = [res_map.get(p, float("inf")) for p in board_key]

            board_list = list(board_key)
            for idx in range(size):
                if board_key[idx] != opponent_piece:
                    continue
                for partner, c1, c2 in geom["bridge_links"][idx]:
                    if board_key[partner] == opponent_piece and board_key[c1] == "." and board_key[c2] == ".":
                        resistances[c1] = float("inf")
                        resistances[c2] = float("inf")

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
                    
                    # Si l'ennemi n'a pas bloqué le pont
                    if board_key[c1] != opponent_piece and board_key[c2] != opponent_piece:
                        pair = (min(idx, partner), max(idx, partner))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        # --- CONNEXION COMPLÈTE ---
                        # Une valeur extrême agit comme un fil parfait (court-circuit)
                        # sans faire planter les mathématiques de la matrice.
                        bridge_c = 1000.0

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
            print(f"Amperage calculation failed: {e}")
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result