

import heapq
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from game_state_hex import GameStateHex
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction

class UnionFind:
    """
    DSU path-compressé + union-by-rank.
    Deux nœuds virtuels : size = start-side, size+1 = end-side.
    Victoire : find(size) == find(size+1).
    """
    __slots__ = ("parent", "rank")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank   = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def clone(self) -> "UnionFind":
        uf = UnionFind.__new__(UnionFind)
        uf.parent = self.parent.copy()
        uf.rank   = self.rank.copy()
        return uf

_GEOM_CACHE: Dict[int, dict] = {}

def _get_geom(n: int) -> dict:
    """
    Précalcule et met en cache toutes les données géométriques pour un plateau n×n :
      neighbors[i]    – indices adjacents de la cellule i
      rows[i]/cols[i] – ligne/colonne de i
      top/bottom/left/right – cellules sur chaque bord
      bridge_pairs[i] – triplets (carrier1, carrier2, partner) de ponts virtuels
    """
    if n in _GEOM_CACHE:
        return _GEOM_CACHE[n]

    DIRS = ((0,-1),(-1,0),(-1,1),(0,1),(1,0),(1,-1))
    size = n * n
    neighbors: List[List[int]] = [[] for _ in range(size)]
    rows = [0] * size
    cols = [0] * size
    top, bottom, left, right = [], [], [], []

    for r in range(n):
        for c in range(n):
            idx = r * n + c
            rows[idx] = r
            cols[idx] = c
            if r == 0:     top.append(idx)
            if r == n - 1: bottom.append(idx)
            if c == 0:     left.append(idx)
            if c == n - 1: right.append(idx)
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    neighbors[idx].append(nr * n + nc)

    # Templates de ponts virtuels (two-bridge) :
    # Paire (A, B) reliée par porteurs C1, C2 — si l'adversaire joue C1, jouer C2 maintient la connexion.
    BRIDGE_TEMPLATES = [
        ( 1,-2,  0,-1,  1,-1),
        ( 2,-1,  1,-1,  1, 0),
        ( 1, 1,  0, 1,  1, 0),
        (-1, 2,  0, 1, -1, 1),
        (-2, 1, -1, 0, -1, 1),
        (-1,-1, -1, 0,  0,-1),
    ]
    bridge_pairs: List[List[Tuple[int,int,int]]] = [[] for _ in range(size)]
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            for pdr,pdc,c1dr,c1dc,c2dr,c2dc in BRIDGE_TEMPLATES:
                pr,pc   = r+pdr, c+pdc
                c1r,c1c = r+c1dr, c+c1dc
                c2r,c2c = r+c2dr, c+c2dc
                if (0<=pr<n and 0<=pc<n and
                        0<=c1r<n and 0<=c1c<n and
                        0<=c2r<n and 0<=c2c<n):
                    bridge_pairs[idx].append((c1r*n+c1c, c2r*n+c2c, pr*n+pc))

    geom = {
        "n": n, "size": size,
        "neighbors": neighbors,
        "rows": rows, "cols": cols,
        "top": top, "bottom": bottom,
        "left": left, "right": right,
        "center": (n - 1) / 2.0,
        "bridge_pairs": bridge_pairs,
    }
    _GEOM_CACHE[n] = geom
    return geom


def _other(piece: str) -> str:
    return "B" if piece == "R" else "R"


def _local_score(board: List[str], idx: int,
                 piece: str, opp: str, geom: dict) -> float:
    """
    Score heuristique O(1) pour une cellule :
      +3.4 par voisin ami  (connectivité)
      +4.6 par voisin ennemi (blocage)
      +0.20 biais centre
      +0.28 progrès vers le bord cible
    """
    own_n = opp_n = 0
    for nb in geom["neighbors"][idx]:
        c = board[nb]
        if c == piece: own_n += 1
        elif c == opp:  opp_n += 1

    center = geom["center"]
    r, c_ = geom["rows"][idx], geom["cols"][idx]
    center_bias = -(abs(r - center) + abs(c_ - center))
    progress = -abs(c_ - center) if piece == "R" else -abs(r - center)
    return 3.4*own_n + 4.6*opp_n + 0.20*center_bias + 0.28*progress


def _connection_cost(board: List[str], piece: str, geom: dict) -> float:
    """
    Dijkstra : coût minimum pour relier les deux bords de 'piece'.
    Coût : 0 si propre pièce, 1 si vide, INF si adversaire.
    """
    n = geom["n"]
    INF = 10**9
    neighbors = geom["neighbors"]

    if piece == "R":
        starts    = geom["top"]
        goal_test = lambda i: geom["rows"][i] == n - 1
    else:
        starts    = geom["left"]
        goal_test = lambda i: geom["cols"][i] == n - 1

    dist = [float(INF)] * (n * n)
    pq: List[Tuple[float, int]] = []

    for s in starts:
        cell = board[s]
        if cell == piece:   w = 0.0
        elif cell == ".":   w = 1.0
        else:               continue
        if w < dist[s]:
            dist[s] = w
            heapq.heappush(pq, (w, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if goal_test(u): return d
        for v in neighbors[u]:
            cv = board[v]
            if cv == piece:   wv = 0.0
            elif cv == ".":   wv = 1.0
            else:              continue
            nd = d + wv
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return float(INF)


def _shortest_path(board: List[str], piece: str,
                   geom: dict) -> Tuple[float, List[int]]:
    """
    Dijkstra avec reconstruction du chemin.
    Retourne (coût, liste de cellules sur le chemin optimal).
    """
    n = geom["n"]
    INF = 10**9
    neighbors = geom["neighbors"]

    if piece == "R":
        starts    = geom["top"]
        goal_test = lambda i: geom["rows"][i] == n - 1
    else:
        starts    = geom["left"]
        goal_test = lambda i: geom["cols"][i] == n - 1

    dist = [float(INF)] * (n * n)
    pred = [-1] * (n * n)
    pq: List[Tuple[float, int]] = []

    for s in starts:
        cell = board[s]
        if cell == piece:   w = 0.0
        elif cell == ".":   w = 1.0
        else:               continue
        if w < dist[s]:
            dist[s] = w
            heapq.heappush(pq, (w, s))

    goal = -1
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if goal_test(u):
            goal = u
            break
        for v in neighbors[u]:
            cv = board[v]
            if cv == piece:   wv = 0.0
            elif cv == ".":   wv = 1.0
            else:              continue
            nd = d + wv
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))

    if goal == -1:
        return float(INF), []

    path = []
    cur = goal
    while cur != -1:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return dist[goal], path


def _has_won(board: List[str], piece: str, geom: dict) -> bool:
    """Vérification de victoire par BFS (utilisée en dehors des rollouts)."""
    n = geom["n"]
    neighbors = geom["neighbors"]
    visited = bytearray(n * n)
    stack = []

    if piece == "R":
        for s in geom["top"]:
            if board[s] == "R" and not visited[s]:
                stack.append(s); visited[s] = 1
        while stack:
            cur = stack.pop()
            if geom["rows"][cur] == n - 1: return True
            for nb in neighbors[cur]:
                if not visited[nb] and board[nb] == "R":
                    visited[nb] = 1; stack.append(nb)
    else:
        for s in geom["left"]:
            if board[s] == "B" and not visited[s]:
                stack.append(s); visited[s] = 1
        while stack:
            cur = stack.pop()
            if geom["cols"][cur] == n - 1: return True
            for nb in neighbors[cur]:
                if not visited[nb] and board[nb] == "B":
                    visited[nb] = 1; stack.append(nb)
    return False


def _board_eval(board: List[str], piece: str, opp: str, geom: dict) -> float:
    """Évaluation statique : différence de coûts de connexion × 14."""
    INF = 10**8
    my_c  = _connection_cost(board, piece, geom)
    opp_c = _connection_cost(board, opp,   geom)
    if my_c >= INF and opp_c >= INF: return 0.0
    if my_c >= INF:  return -10**6
    if opp_c >= INF: return  10**6
    return 14.0 * (opp_c - my_c)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — UNION-FIND POUR ROLLOUTS RAPIDES
#  (Construction incrémentale depuis un board donné)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_uf(board: List[str], piece: str, geom: dict) -> UnionFind:
    """Construit un UnionFind pour 'piece' à partir de l'état courant."""
    n    = geom["n"]
    size = n * n
    uf   = UnionFind(size + 2)
    for idx, cell in enumerate(board):
        if cell != piece: continue
        r, c = geom["rows"][idx], geom["cols"][idx]
        for nb in geom["neighbors"][idx]:
            if board[nb] == piece:
                uf.union(idx, nb)
        if piece == "R":
            if r == 0:     uf.union(idx, size)
            if r == n - 1: uf.union(idx, size + 1)
        else:
            if c == 0:     uf.union(idx, size)
            if c == n - 1: uf.union(idx, size + 1)
    return uf


def _uf_place(uf: UnionFind, board: List[str], idx: int,
              piece: str, geom: dict) -> None:
    """Met à jour le UnionFind après avoir joué 'piece' en idx."""
    n    = geom["n"]
    size = n * n
    r, c = geom["rows"][idx], geom["cols"][idx]
    for nb in geom["neighbors"][idx]:
        if board[nb] == piece:
            uf.union(idx, nb)
    if piece == "R":
        if r == 0:     uf.union(idx, size)
        if r == n - 1: uf.union(idx, size + 1)
    else:
        if c == 0:     uf.union(idx, size)
        if c == n - 1: uf.union(idx, size + 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — ROLLOUT RAPIDE (Union-Find, ~6 000/sec sur 11×11)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_rollout(
    board_tpl: List[str],
    empties_tpl: List[int],
    uf_r_tpl: UnionFind,
    uf_b_tpl: UnionFind,
    first_to_play: str,
    my_piece: str,
    opp_piece: str,
    geom: dict,
    rng: random.Random,
) -> str:
    """
    Rollout optimisé :
      - Copie légère du board et des UF
      - Mélange des empties avec 20% de biais local-score pour le premier joueur
      - Détection de victoire via UF.connected() en O(α) par coup
    Retourne le vainqueur ("R" ou "B").
    """
    n    = geom["n"]
    size = n * n
    neighbors = geom["neighbors"]
    rows      = geom["rows"]
    cols      = geom["cols"]

    board  = board_tpl.copy()
    order  = empties_tpl.copy()
    uf_r   = uf_r_tpl.clone()
    uf_b   = uf_b_tpl.clone()

    # Mélange avec légère pré-sélection biaisée
    rng.shuffle(order)
    n_swap = max(1, len(order) // 5)
    opp_fp = _other(first_to_play)
    for _ in range(n_swap):
        i = rng.randrange(len(order))
        j = rng.randrange(len(order))
        if i != j:
            si = _local_score(board, order[i], first_to_play, opp_fp, geom)
            sj = _local_score(board, order[j], first_to_play, opp_fp, geom)
            if sj > si:
                order[i], order[j] = order[j], order[i]

    to_play = first_to_play
    for idx in order:
        if board[idx] != ".":
            continue
        board[idx] = to_play
        uf = uf_r if to_play == "R" else uf_b
        _uf_place(uf, board, idx, to_play, geom)
        if uf.connected(size, size + 1):
            return to_play
        to_play = opp_piece if to_play == my_piece else my_piece

    return "R" if uf_r.connected(size, size + 1) else "B"

class MCTSNode:
    """
    Nœud de l'arbre MCTS.

    UCT avec progressive bias :
      Q(s,a) + C · √(ln N / n)  +  W · prior / (n + 1)
       ↑ exploitation  ↑ exploration  ↑ biais prior (décroît avec les visites)
    """
    __slots__ = ("move","visits","wins","prior","children","parent","untried")

    def __init__(self, move: int = -1,
                 parent: Optional["MCTSNode"] = None,
                 prior: float = 0.0) -> None:
        self.move     = move
        self.visits   = 0
        self.wins     = 0.0
        self.prior    = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.parent   = parent
        self.untried:  Optional[List[int]] = None

    def uct(self, parent_visits: int, C: float, W: float) -> float:
        if self.visits == 0:
            return 1e18 + self.prior
        exploit = self.wins / self.visits
        explore = C * math.sqrt(math.log(parent_visits) / self.visits)
        bias    = W * self.prior / (self.visits + 1)
        return exploit + explore + bias

    def best_child(self, C: float, W: float) -> "MCTSNode":
        pv = max(self.visits, 1)
        return max(self.children.values(), key=lambda c: c.uct(pv, C, W))

    def most_visited(self) -> "MCTSNode":
        return max(self.children.values(), key=lambda c: c.visits)

    def is_fully_expanded(self) -> bool:
        return self.untried is not None and len(self.untried) == 0

class _MCTS:
    """
    Moteur MCTS avec :
      - Sélection UCT + progressive bias
      - Expansion ordonnée par priors (au plus MAX_CHILDREN enfants)
      - Simulation via _run_rollout (~6 000/sec)
      - Backpropagation avec inversion de récompense
      - Réutilisation de l'arbre entre les tours
    """

    MAX_CHILDREN = 14
    C = 1.15   # constante d'exploration UCT
    W = 0.30   # poids du progressive bias

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def search(
        self,
        board: List[str],
        empties: List[int],
        candidates: List[int],
        my_piece: str,
        opp_piece: str,
        geom: dict,
        time_budget: float,
        priors: Dict[int, float],
        root: Optional[MCTSNode] = None,
    ) -> Tuple[int, MCTSNode]:
        """
        Lance MCTS pendant `time_budget` secondes.
        Ne cherche que parmi `candidates` (déjà filtrés et ordonnés).
        Retourne (meilleur coup, nœud racine).
        """
        deadline = time.perf_counter() + time_budget

        if root is None:
            root = MCTSNode(move=-1)

        # Initialise la liste des coups non essayés à la racine
        if root.untried is None:
            ordered = sorted(candidates,
                             key=lambda m: priors.get(m, 0.0), reverse=True)
            root.untried = ordered[: self.MAX_CHILDREN]

        # Pré-construit les Union-Find de base (clonés pour chaque rollout)
        base_uf_r = _build_uf(board, "R", geom)
        base_uf_b = _build_uf(board, "B", geom)

        # Pré-crée les templates de rollout pour chaque candidat
        # (board+empties après le premier coup joué)
        templates: Dict[int, Tuple[List[str], List[int], UnionFind, UnionFind]] = {}
        for mv in root.untried:
            b = board.copy()
            b[mv] = my_piece
            e = [i for i in empties if i != mv]
            uf_r = base_uf_r.clone()
            uf_b = base_uf_b.clone()
            if my_piece == "R":
                _uf_place(uf_r, b, mv, "R", geom)
            else:
                _uf_place(uf_b, b, mv, "B", geom)
            uf_check = uf_r if my_piece == "R" else uf_b
            if uf_check.connected(geom["n"] * geom["n"],
                                  geom["n"] * geom["n"] + 1):
                return mv, root
            templates[mv] = (b, e, uf_r, uf_b)

        # ── Boucle principale MCTS ───────────────────────────────────────────
        while time.perf_counter() < deadline:
            node, board_at_leaf, empties_at_leaf, uf_r_leaf, uf_b_leaf, depth = \
                self._select_and_expand(
                    root, board, empties, my_piece, opp_piece,
                    geom, priors, base_uf_r, base_uf_b, templates
                )

            to_play_leaf = my_piece if depth % 2 == 0 else opp_piece

            winner = _run_rollout(
                board_at_leaf, empties_at_leaf,
                uf_r_leaf, uf_b_leaf,
                to_play_leaf, my_piece, opp_piece,
                geom, self._rng,
            )

            reward = 1.0 if winner == my_piece else 0.0
            self._backprop(node, reward)

        if not root.children:
            return candidates[0], root
        return root.most_visited().move, root

    def _select_and_expand(
        self,
        root: MCTSNode,
        board: List[str],
        empties: List[int],
        my_piece: str,
        opp_piece: str,
        geom: dict,
        root_priors: Dict[int, float],
        base_uf_r: UnionFind,
        base_uf_b: UnionFind,
        templates: Dict[int, Tuple],
    ) -> Tuple[MCTSNode, List[str], List[int], UnionFind, UnionFind, int]:
        """
        Sélectionne un nœud feuille via UCT, l'expand si possible.
        Retourne (nœud, board à ce nœud, empties, uf_r, uf_b, profondeur).
        """
        n    = geom["n"]
        size = n * n
        node = root
        depth = 0

        # ── Sélection depuis la racine ───────────────────────────────────────
        # Pour les nœuds de profondeur 1 (enfants directs de la racine),
        # on utilise directement les templates pré-calculés.
        if node.is_fully_expanded() and node.children:
            child = node.best_child(self.C, self.W)
            mv = child.move
            if mv in templates:
                b, e, uf_r, uf_b = templates[mv]
                # Descente plus profonde dans l'arbre
                board_cur  = b.copy()
                empties_cur = e.copy()
                uf_r_cur   = uf_r.clone()
                uf_b_cur   = uf_b.clone()
                node = child
                depth = 1
                to_play = opp_piece  # après notre coup, c'est l'adversaire

                while node.is_fully_expanded() and node.children and empties_cur:
                    node = node.best_child(self.C, self.W)
                    mv2 = node.move
                    board_cur[mv2] = to_play
                    uf2 = uf_r_cur if to_play == "R" else uf_b_cur
                    _uf_place(uf2, board_cur, mv2, to_play, geom)
                    # Swap-remove de empties_cur
                    try:
                        empties_cur.remove(mv2)
                    except ValueError:
                        pass
                    depth += 1
                    to_play = opp_piece if to_play == my_piece else my_piece

                # Expansion du nœud courant si nécessaire
                if not node.is_fully_expanded() and empties_cur:
                    if node.untried is None:
                        cur_priors = {i: _local_score(board_cur, i, to_play,
                                                      _other(to_play), geom)
                                      for i in empties_cur}
                        node.untried = sorted(
                            empties_cur,
                            key=lambda m: cur_priors.get(m, 0.0), reverse=True
                        )[: self.MAX_CHILDREN]

                    if node.untried:
                        mv3 = node.untried.pop(0)
                        prior3 = root_priors.get(mv3, 0.0)
                        child3 = MCTSNode(move=mv3, parent=node, prior=prior3)
                        node.children[mv3] = child3

                        board_cur[mv3] = to_play
                        uf3 = uf_r_cur if to_play == "R" else uf_b_cur
                        _uf_place(uf3, board_cur, mv3, to_play, geom)
                        try:
                            empties_cur.remove(mv3)
                        except ValueError:
                            pass
                        depth += 1
                        node = child3

                return node, board_cur, empties_cur, uf_r_cur, uf_b_cur, depth

        # ── Expansion depuis la racine ───────────────────────────────────────
        if node.untried is None:
            ordered = sorted(empties,
                             key=lambda m: root_priors.get(m, 0.0), reverse=True)
            node.untried = ordered[: self.MAX_CHILDREN]

        if node.untried:
            mv = node.untried.pop(0)
            prior = root_priors.get(mv, 0.0)
            child = MCTSNode(move=mv, parent=node, prior=prior)
            node.children[mv] = child

            if mv in templates:
                b, e, uf_r, uf_b = templates[mv]
                return child, b.copy(), e.copy(), uf_r.clone(), uf_b.clone(), 1

            b = board.copy()
            b[mv] = my_piece
            e = [i for i in empties if i != mv]
            uf_r = base_uf_r.clone()
            uf_b = base_uf_b.clone()
            uf_t = uf_r if my_piece == "R" else uf_b
            _uf_place(uf_t, b, mv, my_piece, geom)
            return child, b, e, uf_r, uf_b, 1

        # Feuille terminale (plus d'empties)
        return node, board.copy(), [], base_uf_r.clone(), base_uf_b.clone(), 0

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        """Propage wins/visits de la feuille à la racine en alternant la récompense."""
        cur, r = node, reward
        while cur is not None:
            cur.visits += 1
            cur.wins   += r
            r   = 1.0 - r
            cur = cur.parent


class MyPlayer(PlayerHex):
    EXACT_SOLVE_LIMIT    = 8     # solver exact si ≤ N vides
    PRESELECT_LIMIT      = 28    # candidats pré-sélectionnés par score local
    CANDIDATE_LIMIT      = 14    # candidats finaux pour MCTS
    SHALLOW_SEARCH_WIDTH = 8     # largeur du minimax shallow
    SHALLOW_MID_DEPTH    = 2     # profondeur shallow en milieu de partie
    SHALLOW_LATE_DEPTH   = 3     # profondeur shallow en fin de partie
    TIME_MARGIN          = 0.12  # réserve de temps par tour (secondes)

    def __init__(self, piece_type: str, name: str = "FusedHexBot") -> None:
        super().__init__(piece_type, name)
        self._rng   = random.Random(42)
        self._mcts  = _MCTS(self._rng)
        self._root: Optional[MCTSNode] = None   # réutilisation de l'arbre

    def compute_action(
        self,
        current_state: GameStateHex,
        remaining_time: float = 15 * 60,
        **kwargs,
    ) -> Action:
        del kwargs
        t0 = time.perf_counter()
        n  = current_state.get_rep().get_dimensions()[0]
        geom = _get_geom(n)
        board, empties = self._extract_board(current_state, n)

        if not empties:
            return self._to_action(0, n)

        my  = self.piece_type
        opp = _other(my)

        opening = self._opening_move(board, empties, my, n, geom)
        if opening is not None:
            self._root = None
            return self._to_action(opening, n)

        win = self._find_immediate_win(board, empties, my, geom)
        if win is not None:
            self._root = None
            return self._to_action(win, n)

        threats = self._find_forced_blocks(board, empties, opp, geom)
        if threats:
            block = self._best_local_move(board, threats, my, opp, geom)
            self._root = None
            return self._to_action(block, n)

        anchor = self._early_anchor_move(board, empties, my, opp, geom)
        if anchor is not None:
            self._root = None
            return self._to_action(anchor, n)

        if len(empties) > self.EXACT_SOLVE_LIMIT:
            race = self._race_path_move(board, empties, my, opp, geom)
            if race is not None:
                self._root = None
                return self._to_action(race, n)

        crit = self._find_critical_block(board, empties, my, opp, geom)
        if crit is not None:
            self._root = None
            return self._to_action(crit, n)

        if len(empties) <= self.EXACT_SOLVE_LIMIT:
            exact = self._solve_exact_root(board, empties, my, geom)
            if exact is not None:
                self._root = None
                return self._to_action(exact, n)

        candidates = self._select_candidates(board, empties, my, opp, geom)
        if not candidates:
            return self._to_action(empties[0], n)

        candidates, tactical_priors = self._tactical_filter(
            board, empties, candidates, my, opp, geom
        )
        if len(candidates) == 1:
            self._root = None
            return self._to_action(candidates[0], n)

        budget  = self._time_budget(remaining_time, len(empties))
        elapsed = time.perf_counter() - t0
        budget  = max(0.3, budget - elapsed)

        shallow_priors: Dict[int, float] = {}
        if budget > 0.8 and len(empties) <= 80:
            depth = (self.SHALLOW_LATE_DEPTH
                     if len(empties) <= 30 and remaining_time > 50.0
                     else self.SHALLOW_MID_DEPTH)
            shallow_window = min(1.1, 0.50 * budget)

            shallow_priors = self._compute_shallow_priors(
                board, empties, candidates, my, opp, geom,
                depth=depth,
                width=self.SHALLOW_SEARCH_WIDTH,
                time_limit=shallow_window,
            )
            if shallow_priors:
                candidates = sorted(candidates,
                                    key=lambda m: shallow_priors.get(m, -1e9),
                                    reverse=True)[: self.CANDIDATE_LIMIT]

        merged_priors = tactical_priors.copy()
        merged_priors.update(shallow_priors)

        _, my_path  = _shortest_path(board, my,  geom)
        _, opp_path = _shortest_path(board, opp, geom)
        my_set  = set(my_path)
        opp_set = set(opp_path)
        bridge_threats = self._find_bridge_threats(board, my, geom)
        bridge_set = set(bridge_threats)
        for mv in candidates:
            bonus = 0.0
            if mv in my_set:   bonus += 7.0
            if mv in opp_set:  bonus += 6.0
            if mv in bridge_set: bonus += 18.0
            merged_priors[mv] = merged_priors.get(mv, 0.0) + bonus

        elapsed = time.perf_counter() - t0
        mcts_budget = max(0.25, budget - (time.perf_counter() - t0 - elapsed))
        mcts_budget = max(0.25, self._time_budget(remaining_time, len(empties))
                          - (time.perf_counter() - t0))

        best_move, new_root = self._mcts.search(
            board = board,
            empties = empties,
            candidates = candidates,
            my_piece = my,
            opp_piece = opp,
            geom = geom,
            time_budget= mcts_budget,
            priors = merged_priors,
            root = self._root,
        )

        self._root = new_root
        return self._to_action(best_move, n)

    def _time_budget(self, remaining: float, nb_empties: int) -> float:
        """
        Allocation adaptative :
          ~3.2% du temps restant, capé selon la phase de jeu.
        """
        base = min(4.5, max(0.5, remaining * 0.032))
        if   nb_empties > 100: base = min(base, 1.0)
        elif nb_empties > 70:  base = min(base, 1.8)
        elif nb_empties > 40:  base = min(base, 3.0)
        return base - self.TIME_MARGIN


    def _to_action(self, idx: int, n: int) -> StatelessAction:
        return StatelessAction(
            {"piece": self.piece_type, "position": (idx // n, idx % n)}
        )

    def _extract_board(self, state: GameStateHex, n: int) -> Tuple[List[str], List[int]]:
        board = ["."] * (n * n)
        for (r, c), piece in state.get_rep().get_env().items():
            board[r * n + c] = piece.get_type()
        empties = [i for i, cell in enumerate(board) if cell == "."]
        return board, empties


    def _opening_move(self, board: List[str], empties: List[int],
                      piece: str, n: int, geom: dict) -> Optional[int]:
        """Joue le centre (ou quasi-centre) sur le propre axe lors des 2 premiers coups."""
        if len(empties) < n * n - 2:
            return None
        center = n // 2
        if piece == "R":
            candidates = [center*n+center, center*n+center+1, center*n+center-1]
        else:
            candidates = [(center+1)*n+center, center*n+center, (center-1)*n+center]
        eset = set(empties)
        for mv in candidates:
            if 0 <= mv < n*n and mv in eset:
                return mv
        return None


    def _find_immediate_win(self, board: List[str], empties: List[int],
                            piece: str, geom: dict) -> Optional[int]:
        for idx in empties:
            board[idx] = piece
            won = _has_won(board, piece, geom)
            board[idx] = "."
            if won:
                return idx
        return None

    def _find_forced_blocks(self, board: List[str], empties: List[int],
                             opp: str, geom: dict) -> List[int]:
        threats = []
        for idx in empties:
            board[idx] = opp
            won = _has_won(board, opp, geom)
            board[idx] = "."
            if won:
                threats.append(idx)
        return threats

    def _best_local_move(self, board: List[str], moves: List[int],
                          piece: str, opp: str, geom: dict) -> int:
        best_mv, best_sc = moves[0], -1e18
        for idx in moves:
            sc = _local_score(board, idx, piece, opp, geom)
            if sc > best_sc:
                best_sc = sc
                best_mv = idx
        return best_mv

    def _count_immediate_wins(self, board: List[str], empties: List[int],
                               piece: str, geom: dict, cap: int = 99) -> int:
        count = 0
        for idx in empties:
            board[idx] = piece
            won = _has_won(board, piece, geom)
            board[idx] = "."
            if won:
                count += 1
                if count >= cap:
                    return count
        return count


    def _early_anchor_move(self, board: List[str], empties: List[int],
                            piece: str, opp: str, geom: dict) -> Optional[int]:
        """
        En début de partie, ancre une pièce sur le bord de départ si absent.
        Favorise le bord centré pour maximiser les options de connexion.
        """
        n = geom["n"]
        if len(empties) < n * n - 26:
            return None

        edge = geom["top"] if piece == "R" else geom["left"]
        if any(board[i] == piece for i in edge):
            return None

        edge_empties = [i for i in edge if board[i] == "."]
        if not edge_empties:
            return None

        center = geom["center"]
        best_i, best_sc = edge_empties[0], -1e18
        for idx in edge_empties:
            local = _local_score(board, idx, piece, opp, geom)
            edge_center_bias = -(abs(geom["cols"][idx] - center)
                                 if piece == "R"
                                 else abs(geom["rows"][idx] - center))
            sc = local + 0.9 * edge_center_bias
            if sc > best_sc:
                best_sc = sc
                best_i  = idx
        return best_i


    def _race_path_move(self, board: List[str], empties: List[int],
                        piece: str, opp: str, geom: dict) -> Optional[int]:
        """
        Compare les coûts de connexion Dijkstra des deux joueurs.
        Si l'adversaire est en avance, bloque son chemin ; sinon, avance sur le sien.
        """
        my_cost,  my_path  = _shortest_path(board, piece, geom)
        opp_cost, opp_path = _shortest_path(board, opp,   geom)

        my_empties  = [i for i in my_path  if board[i] == "."]
        opp_empties = [i for i in opp_path if board[i] == "."]
        if not my_empties and not opp_empties:
            return None

        need_block = opp_cost <= my_cost + 0.35
        target     = opp_empties if need_block and opp_empties else my_empties
        if not target:
            target = opp_empties if opp_empties else my_empties
        if not target:
            return None

        best_mv, best_sc = target[0], -1e18
        for mv in target:
            board[mv] = piece
            ma = _connection_cost(board, piece, geom)
            oa = _connection_cost(board, opp,   geom)
            board[mv] = "."
            gain_my    = my_cost  - ma
            gain_block = oa       - opp_cost
            local      = _local_score(board, mv, piece, opp, geom)
            sc = (9.0 * gain_block + 2.4 * gain_my + 0.35 * local
                  if need_block
                  else 9.0 * gain_my + 2.4 * gain_block + 0.35 * local)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
        return best_mv


    def _find_critical_block(self, board: List[str], empties: List[int],
                              piece: str, opp: str, geom: dict) -> Optional[int]:
        """
        Détecte si l'adversaire est proche de gagner et choisit le meilleur blocage.
        Ne retourne quelque chose que si la situation est vraiment urgente.
        """
        my_now  = _connection_cost(board, piece, geom)
        opp_now = _connection_cost(board, opp,   geom)

        if opp_now > 3.0 and opp_now >= my_now - 1.0:
            return None

        best_mv, best_sc = None, -1e18
        for idx in empties:
            board[idx] = piece
            opp_after = _connection_cost(board, opp,   geom)
            my_after  = _connection_cost(board, piece, geom)
            board[idx] = "."
            sc = (9.0 * (opp_after - opp_now)
                  + 2.3 * (my_now - my_after)
                  + 0.25 * _local_score(board, idx, piece, opp, geom))
            if sc > best_sc:
                best_sc = sc
                best_mv = idx

        if best_mv is None:
            return None
        if opp_now <= 2.0:
            return best_mv
        if opp_now <= 3.0 and best_sc > 1.0:
            return best_mv
        return None

    def _find_bridge_threats(self, board: List[str],
                              piece: str, geom: dict) -> List[int]:
        """Retourne les porteurs de ponts virtuels menacés par l'adversaire."""
        bridge_pairs = geom["bridge_pairs"]
        threats, seen = [], set()
        for idx, cell in enumerate(board):
            if cell != piece: continue
            for c1, c2, partner in bridge_pairs[idx]:
                if board[partner] != piece: continue
                if board[c1] == "." and board[c2] == piece and c1 not in seen:
                    threats.append(c1); seen.add(c1)
                elif board[c2] == "." and board[c1] == piece and c2 not in seen:
                    threats.append(c2); seen.add(c2)
        return threats

    def _select_candidates(self, board: List[str], empties: List[int],
                            piece: str, opp: str, geom: dict) -> List[int]:
        """
        Réduit les empties à CANDIDATE_LIMIT coups pertinents :
          1. Filtre la frontière (cellules adjacentes à des pièces posées)
          2. Pré-sélectionne par score local
          3. Raffine par gain de coût de connexion Dijkstra
        """
        neighbors = geom["neighbors"]
        frontier  = [i for i in empties
                     if any(board[nb] != "." for nb in neighbors[i])]
        if not frontier:
            frontier = empties.copy()

        scored = sorted(
            ((
                _local_score(board, i, piece, opp, geom),
                i,
            ) for i in frontier),
            reverse=True,
        )
        preselect = [i for _, i in scored[: self.PRESELECT_LIMIT]]

        refined = []
        for idx in preselect:
            board[idx] = piece
            my_c  = _connection_cost(board, piece, geom)
            opp_c = _connection_cost(board, opp,   geom)
            board[idx] = "."
            local    = _local_score(board, idx, piece, opp, geom)
            pressure = opp_c - my_c
            refined.append((6.8 * pressure + local, idx))

        refined.sort(reverse=True)
        return [i for _, i in refined[: self.CANDIDATE_LIMIT]]

    def _tactical_filter(
        self,
        board: List[str],
        empties: List[int],
        candidates: List[int],
        piece: str,
        opp: str,
        geom: dict,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Pour chaque candidat :
          - Vérifie s'il gagne immédiatement = retour direct
          - Compte les menaces adverses créées (coups qui nous donnent une victoire immédiate)
          - Compte nos propres menaces créées
          - Élimine les coups qui créent des double-menaces adverses
        Retourne (candidats filtrés, dict de priors tactiques).
        """
        detailed   = []
        safe_moves = []
        priors: Dict[int, float] = {}

        for mv in candidates:
            board[mv] = piece
            rest = [e for e in empties if e != mv]

            if _has_won(board, piece, geom):
                board[mv] = "."
                return [mv], {mv: 10**6}

            opp_threats = self._count_immediate_wins(board, rest, opp,   geom, cap=3)
            my_threats  = self._count_immediate_wins(board, rest, piece, geom, cap=3)
            local       = _local_score(board, mv, piece, opp, geom)
            pressure    = (_connection_cost(board, opp, geom)
                           - _connection_cost(board, piece, geom))
            board[mv] = "."

            tactical_score = (
                120.0 * my_threats
                - 240.0 * opp_threats
                + 5.0   * pressure
                + local
            )
            priors[mv] = tactical_score
            detailed.append((opp_threats, -my_threats, -tactical_score, mv))
            if opp_threats == 0:
                safe_moves.append(mv)

        detailed.sort()
        ordered_all  = [mv for _, _, _, mv in detailed]
        safe_set     = set(safe_moves)
        ordered_safe = [mv for mv in ordered_all if mv in safe_set]
        return (ordered_safe if ordered_safe else ordered_all), priors

    def _compute_shallow_priors(
        self,
        board: List[str],
        empties: List[int],
        candidates: List[int],
        piece: str,
        opp: str,
        geom: dict,
        depth: int,
        width: int,
        time_limit: float,
    ) -> Dict[int, float]:
        """
        Lance un minimax limité à 'depth' niveaux sur chaque candidat
        pour obtenir des priors de haute qualité (bien meilleur que le simple local score).
        Le budget temps est strict : on s'arrête quand il est épuisé.
        """
        if depth <= 0 or not candidates:
            return {}
        deadline = time.perf_counter() + max(0.05, time_limit)
        priors: Dict[int, float] = {}

        for mv in candidates:
            if time.perf_counter() >= deadline:
                break
            board[mv] = piece
            rest = [e for e in empties if e != mv]
            if _has_won(board, piece, geom):
                board[mv] = "."
                priors[mv] = 10**6
                continue
            score = self._minimax_limited(
                board, rest, opp, piece,
                depth - 1, -1e9, 1e9, width, geom,
            )
            board[mv] = "."
            priors[mv] = score

        return priors

    def _minimax_limited(
        self,
        board: List[str],
        empties: List[int],
        to_play: str,
        root_piece: str,
        depth: int,
        alpha: float,
        beta: float,
        width: int,
        geom: dict,
    ) -> float:
        """Alpha-beta minimax à profondeur limitée avec évaluation Dijkstra."""
        opp_root = _other(root_piece)

        if _has_won(board, root_piece, geom): return  10**6 + depth
        if _has_won(board, opp_root,   geom): return -10**6 - depth
        if depth == 0 or not empties:
            return _board_eval(board, root_piece, opp_root, geom)

        maximizing  = (to_play == root_piece)
        next_player = _other(to_play)
        opp_to      = _other(to_play)
        moves = sorted(
            empties,
            key=lambda i: _local_score(board, i, to_play, opp_to, geom),
            reverse=True,
        )[: width]

        if maximizing:
            best = -1e9
            for mv in moves:
                board[mv] = to_play
                rest = [e for e in empties if e != mv]
                val  = self._minimax_limited(board, rest, next_player,
                                             root_piece, depth-1, alpha, beta, width, geom)
                board[mv] = "."
                if val > best: best = val
                alpha = max(alpha, best)
                if alpha >= beta: break
            return best
        else:
            best = 1e9
            for mv in moves:
                board[mv] = to_play
                rest = [e for e in empties if e != mv]
                val  = self._minimax_limited(board, rest, next_player,
                                             root_piece, depth-1, alpha, beta, width, geom)
                board[mv] = "."
                if val < best: best = val
                beta = min(beta, best)
                if alpha >= beta: break
            return best

    def _solve_exact_root(self, board: List[str], empties: List[int],
                           piece: str, geom: dict) -> Optional[int]:
        """
        Solver exact pour fins de partie (≤ EXACT_SOLVE_LIMIT vides).
        Alpha-beta complet avec table de transposition.
        Retourne le coup gagnant, ou le meilleur coup possible.
        """
        opp  = _other(piece)
        tt: Dict[Tuple[str, str], int] = {}
        move_order = sorted(
            empties,
            key=lambda i: _local_score(board, i, piece, opp, geom),
            reverse=True,
        )
        best_mv, best_sc = None, -2
        alpha, beta = -1, 1

        for idx in move_order:
            board[idx] = piece
            if _has_won(board, piece, geom):
                board[idx] = "."
                return idx
            rest  = [e for e in empties if e != idx]
            score = self._exact_search(board, rest, opp, piece, alpha, beta, tt, geom)
            board[idx] = "."
            if score > best_sc:
                best_sc = score
                best_mv = idx
            alpha = max(alpha, best_sc)
            if alpha >= beta:
                break

        return best_mv

    def _exact_search(
        self,
        board: List[str],
        empties: List[int],
        to_play: str,
        root_piece: str,
        alpha: int,
        beta: int,
        tt: Dict,
        geom: dict,
    ) -> int:
        """Branche récursive du solver exact avec mémorisation."""
        if _has_won(board, root_piece,         geom): return  1
        if _has_won(board, _other(root_piece),  geom): return -1
        if not empties:                                return -1

        key    = ("".join(board), to_play)
        cached = tt.get(key)
        if cached is not None:
            return cached

        opp_tp = _other(to_play)
        moves  = sorted(
            empties,
            key=lambda i: _local_score(board, i, to_play, opp_tp, geom),
            reverse=True,
        )
        maximizing = (to_play == root_piece)

        if maximizing:
            best = -2
            for idx in moves:
                board[idx] = to_play
                rest  = [e for e in empties if e != idx]
                val   = self._exact_search(board, rest, opp_tp, root_piece, alpha, beta, tt, geom)
                board[idx] = "."
                if val > best: best = val
                alpha = max(alpha, best)
                if alpha >= beta: break
        else:
            best = 2
            for idx in moves:
                board[idx] = to_play
                rest  = [e for e in empties if e != idx]
                val   = self._exact_search(board, rest, opp_tp, root_piece, alpha, beta, tt, geom)
                board[idx] = "."
                if val < best: best = val
                beta = min(beta, best)
                if alpha >= beta: break

        tt[key] = best
        return best