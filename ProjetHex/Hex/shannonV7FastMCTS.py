'''
A partir de ShannonV6FastMCTS, on change les points de recompenses. Maintenant on cherche a setendre.'''

from __future__ import annotations

import math
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


def _enemy_bridge_blocked(board: List[str], target_piece: str, geom: dict) -> Set[int]:
    """Cases vides virtuellement contrôlées par un bridge adverse.

    Si l'adversaire a deux pièces en bridge avec les deux intermédiaires
    libres, ces cases sont quasi-impénétrables : jouer sur l'une permet
    à l'adversaire de sceller l'autre. On les traite comme bloquées
    (conductance 0) dans le réseau du joueur `target_piece`.
    """
    opp = _other(target_piece)
    blocked: Set[int] = set()

    for idx in range(len(board)):
        if board[idx] != opp:
            continue
        for partner, c1, c2 in geom["bridge_links"][idx]:
            if board[partner] != opp:
                continue
            if board[c1] == "." and board[c2] == ".":
                blocked.add(c1)
                blocked.add(c2)

    return blocked


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
# Joueur MCTS
# ---------------------------------------------------------------------------

class MyPlayer(PlayerHex):
    WIN_SCORE = 10**7
    MAX_AMPERAGE_CACHE = 50000
    CG_MAXITER = 40
    TOP_K = 12
    TIME_FRACTION = 0.06
    MIN_TIME_PER_MOVE = 0.5
    EXPLORATION = 1.414
    ROLLOUT_DEPTH = 6

    def __init__(self, piece_type: str, name: str = "AmperesMCTS") -> None:
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

        # --- Ouverture ---
        opening = self._opening_move(board, empties, my_piece, geom)
        if opening is not None:
            return self._to_action(opening, n)

        # --- Coups gagnants immédiats ---
        my_wins = self._winning_moves(board, empties, my_piece, geom)
        if my_wins:
            return self._to_action(my_wins[0], n)

        opp_wins = self._winning_moves(board, empties, opp_piece, geom)
        if opp_wins:
            return self._to_action(opp_wins[0], n)

        # --- MCTS ---
        time_budget = max(self.MIN_TIME_PER_MOVE, remaining_time * self.TIME_FRACTION)
        deadline = time.time() + time_budget

        candidates = self._ordered_moves(board, empties, my_piece, geom)[: self.TOP_K]

        root = MCTSNode(
            move=None,
            parent=None,
            untried=list(candidates),
            to_play=my_piece,
        )

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
                mv = node.untried.pop()
                sim_board[mv] = node.to_play
                sim_empties = [e for e in sim_empties if e != mv]
                next_play = _other(node.to_play)

                child_candidates = self._ordered_moves(
                    sim_board, sim_empties, next_play, geom
                )[: self.TOP_K]

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
            best = root.most_visited_child()
            best_move = best.move
        else:
            best_move = candidates[0]

        return self._to_action(best_move, n)

    # -----------------------------------------------------------------------
    # Rollout : quelques coups aléatoires guidés puis évaluation heuristique
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

        if _has_won(sim_board, root_piece, geom):
            return 1.0
        if _has_won(sim_board, _other(root_piece), geom):
            return 0.0

        for _ in range(self.ROLLOUT_DEPTH):
            if not sim_empties:
                break

            winning = None
            for idx in sim_empties:
                sim_board[idx] = current
                if _has_won(sim_board, current, geom):
                    winning = idx
                    sim_board[idx] = "."
                    break
                sim_board[idx] = "."

            if winning is not None:
                return 1.0 if current == root_piece else 0.0

            ordered = self._ordered_moves(sim_board, sim_empties, current, geom)
            top = min(3, len(ordered))
            mv = ordered[np.random.randint(top)]

            sim_board[mv] = current
            sim_empties = [e for e in sim_empties if e != mv]
            current = _other(current)

        my_amp = self._calculate_amperage(tuple(sim_board), root_piece, geom)
        opp_amp = self._calculate_amperage(tuple(sim_board), _other(root_piece), geom)

        diff = my_amp - opp_amp
        return 1.0 / (1.0 + math.exp(-0.05 * diff))

    # -----------------------------------------------------------------------
    # Modèle électrique (identique à la version MiniMax, avec blocage bridges adverses)
    # -----------------------------------------------------------------------

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

        # Cases virtuellement bloquées par les bridges adverses
        blocked = _enemy_bridge_blocked(board_list, target_piece, geom)

        def edge_conductance(cell_i: str, cell_j: str, idx_i: int, idx_j: int) -> float:
            if cell_i == opponent_piece or cell_j == opponent_piece:
                return 0.0
            if idx_i in blocked or idx_j in blocked:
                return 0.0
            if cell_i == target_piece and cell_j == target_piece:
                return 1000.0
            if cell_i == target_piece or cell_j == target_piece:
                return 2.0
            return 1.0

        def source_conductance(cell: str, idx: int) -> float:
            if cell == opponent_piece or idx in blocked:
                return 0.0
            if cell == target_piece:
                return 1000.0
            return 1.0

        g_mat = lil_matrix((size, size))
        current = np.zeros(size)
        diag_extra = np.zeros(size)

        for idx in range(size):
            cell_i = board_key[idx]

            if cell_i == opponent_piece or idx in blocked:
                g_mat[idx, idx] = 1.0
                continue

            diag_sum = 0.0
            for nb in geom["neighbors"][idx]:
                cell_j = board_key[nb]
                c = edge_conductance(cell_i, cell_j, idx, nb)
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
                    source_c = source_conductance(cell_i, idx)
                if row == n - 1:
                    sink_c = source_conductance(cell_i, idx)
            else:
                if col == 0:
                    source_c = source_conductance(cell_i, idx)
                if col == n - 1:
                    sink_c = source_conductance(cell_i, idx)

            g_mat[idx, idx] = diag_sum + source_c + sink_c
            current[idx] = source_c

        # Arêtes virtuelles pour les bridges alliés
        vconns = _virtual_connections(board_list, target_piece, geom)
        seen_pairs: Set[Tuple[int, int]] = set()

        for idx, partner, c1, c2 in vconns:
            pair = (min(idx, partner), max(idx, partner))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            v1 = board_key[c1]
            v2 = board_key[c2]

            if v1 == "." and v2 == ".":
                bridge_c = 400.0
            elif v1 == target_piece or v2 == target_piece:
                bridge_c = 800.0
            else:
                bridge_c = 100.0

            g_mat[idx, partner] = g_mat[idx, partner] - bridge_c
            g_mat[partner, idx] = g_mat[partner, idx] - bridge_c
            diag_extra[idx] += bridge_c
            diag_extra[partner] += bridge_c

        for idx in range(size):
            if diag_extra[idx] != 0.0 and board_key[idx] != opponent_piece and idx not in blocked:
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
                if cell == opponent_piece or idx in blocked:
                    continue
                sc = source_conductance(cell, idx)
                total_current += sc * (1.0 - voltages[idx])
            result = float(total_current)
        except np.linalg.LinAlgError:
            result = 0.0

        self._amperage_cache[cache_key] = result
        if len(self._amperage_cache) > self.MAX_AMPERAGE_CACHE:
            self._amperage_cache.popitem(last=False)
        return result

    # -----------------------------------------------------------------------
    # Ordonnancement (identique)
    # -----------------------------------------------------------------------

    def _ordered_moves(self, board: List[str], moves: List[int], piece: str, geom: dict) -> List[int]:
        opp = _other(piece)
        rescue_moves = self._bridge_responses(board, piece, geom)
        cut_moves = self._bridge_cut_targets(board, opp, geom)
        n = geom["n"]
        rows = geom["rows"]
        cols = geom["cols"]
        center = geom["center"]

        # Calculer la portée actuelle le long de l'axe objectif
        my_min = n
        my_max = -1
        for i in range(len(board)):
            if board[i] == piece:
                coord = rows[i] if piece == "R" else cols[i]
                if coord < my_min:
                    my_min = coord
                if coord > my_max:
                    my_max = coord

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

            row = rows[idx]
            col = cols[idx]

            # Progression vers les bords cibles
            goal_coord = row if piece == "R" else col
            if my_max >= 0:
                extend_bonus = 0.0
                if goal_coord < my_min:
                    extend_bonus = 6.0 * (my_min - goal_coord)
                elif goal_coord > my_max:
                    extend_bonus = 6.0 * (goal_coord - my_max)
            else:
                extend_bonus = 0.0

            # Proximité aux bords cibles
            dist_to_start = goal_coord
            dist_to_end = (n - 1) - goal_coord
            edge_bonus = 2.0 * (1.0 / (1.0 + min(dist_to_start, dist_to_end)))

            # Centre le long de l'axe secondaire
            secondary_coord = col if piece == "R" else row
            secondary_center = -(abs(secondary_coord - center)) * 0.4

            # Bridge / espace
            bridge_bonus = self._bridge_build_bonus(board, idx, piece, geom)
            if idx in rescue_moves:
                bridge_bonus += 18.0
            if idx in cut_moves:
                bridge_bonus += 5.0

            # Reach bonus : prise d'espace via bridge distance
            reach_bonus = 0.0
            if own_n == 0:
                for partner, c1, c2 in geom["bridge_links"][idx]:
                    if board[partner] == piece:
                        v1 = board[c1]
                        v2 = board[c2]
                        if v1 != opp and v2 != opp:
                            reach_bonus = 5.0
                            break

            score = (
                1.0 * own_n
                + 2.0 * opp_n
                + secondary_center
                + extend_bonus
                + edge_bonus
                + reach_bonus
                + bridge_bonus
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

    # -----------------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------------

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