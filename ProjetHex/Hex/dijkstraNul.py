import time
import heapq
from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex


class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "HexAlgBot"):
        super().__init__(piece_type, name)
        self.memo = {}
        self.board_dim = None

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 900, **kwargs) -> Action:
        self.memo.clear()
        self.board_dim = current_state.get_rep().get_dimensions()
        start_time = time.time()
        best_action = None
        
        # Budget fraction of remaining time, max 4.0 seconds to prevent timeouts
        time_budget = min(remaining_time / 15.0, 4.0)

        # Iterative Deepening
        for depth in range(1, 10):
            if time.time() - start_time > time_budget:
                break
            
            val, action = self._minimax(
                current_state, depth,
                float('-inf'), float('inf'),
                True, start_time, time_budget, depth
            )
            
            if action:
                best_action = action

        if best_action is None:
            # Fallback if depth 1 fails (extremely rare)
            actions = list(current_state.generate_possible_stateful_actions())
            actions = self._order_moves(actions, current_state)
            best_action = actions[0]

        return current_state.convert_stateful_action_to_stateless_action(best_action)

    def _minimax(self, state, depth, alpha, beta, maximizing, start_time, budget, initial_depth):
        if time.time() - start_time > budget:
            return self._evaluate(state, depth), None

        if state.is_done() or depth == 0:
            return self._evaluate(state, depth), None

        actions = list(state.generate_possible_stateful_actions())
        actions = self._order_moves(actions, state)

        best_action = None
        if maximizing:
            best_val = float('-inf')
            for action in actions:
                val, _ = self._minimax(
                    action.get_next_game_state(), depth - 1,
                    alpha, beta, False, start_time, budget, initial_depth
                )
                if val > best_val:
                    best_val, best_action = val, action
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            for action in actions:
                val, _ = self._minimax(
                    action.get_next_game_state(), depth - 1,
                    alpha, beta, True, start_time, budget, initial_depth
                )
                if val < best_val:
                    best_val, best_action = val, action
                beta = min(beta, val)
                if beta <= alpha:
                    break

        return best_val, best_action

    def _order_moves(self, actions, state):
        """
        Orders moves by Shortest-Path critical points. 
        Switches to Defense Mode if the opponent is ahead.
        """
        env = state.get_rep().get_env()
        ci, cj = self.board_dim[0] // 2, self.board_dim[1] // 2
        opp = 'B' if self.piece_type == 'R' else 'R'
        
        our_cells = {pos for pos, p in env.items() if p.get_type() == self.piece_type}
        opp_cells = {pos for pos, p in env.items() if p.get_type() != self.piece_type}

        # Find the critical path cells for both players
        my_cost, my_path = self._dijkstra_path(env, self.piece_type)
        opp_cost, opp_path = self._dijkstra_path(env, opp)
        
        my_path_set = set(my_path)
        opp_path_set = set(opp_path)

        def priority(action):
            next_env = action.get_next_game_state().get_rep().get_env()
            new = set(next_env.keys()) - set(env.keys())
            if not new:
                return (999, 999)
            i, j = next(iter(new))
            
            score = 0
            
            # 1. Critical Paths: Defense vs Attack Priority
            if opp_cost <= my_cost + 1:
                # DEFENSE MODE: Opponent is winning or tied, prioritize blocking!
                if (i, j) in opp_path_set: score += 2000
                if (i, j) in my_path_set:  score += 1000
            else:
                # ATTACK MODE: We are ahead, prioritize connecting our path!
                if (i, j) in my_path_set:  score += 2000
                if (i, j) in opp_path_set: score += 1000
                
            # 2. Structural Heuristics
            bridges = sum(1 for bi, bj in self._bridges(i, j) if (bi, bj) in our_cells)
            opp_adj = sum(1 for ni, nj in self._neighbors(i, j) if (ni, nj) in opp_cells)
            opp_bridges_blocked = sum(1 for bi, bj in self._bridges(i, j) if (bi, bj) in opp_cells)
            our_adj = sum(1 for ni, nj in self._neighbors(i, j) if (ni, nj) in our_cells)
            
            score += (bridges * 5) + (opp_bridges_blocked * 4) + (opp_adj * 3) - (our_adj * 2)
            
            center_dist = abs(i - ci) + abs(j - cj)
            
            return (-score, center_dist)

        return sorted(actions, key=priority)

    def _evaluate(self, state, current_depth) -> float:
        env = state.get_rep().get_env()
        opp = 'B' if self.piece_type == 'R' else 'R'
        
        # Caching Dijkstra to save massive amounts of time
        key_list = sorted((k[0], k[1], v.get_type()) for k, v in env.items())
        key = hash(tuple(key_list))
        
        if key in self.memo:
            my_cost, opp_cost = self.memo[key]
        else:
            my_cost, _ = self._dijkstra_path(env, self.piece_type)
            opp_cost, _ = self._dijkstra_path(env, opp)
            self.memo[key] = (my_cost, opp_cost)

        # Base evaluations
        if my_cost == 0:
            return 10000.0 + current_depth
        elif opp_cost == 0:
            return -10000.0 - current_depth
            
        score = (100.0 / my_cost) - (100.0 / opp_cost)

        # Structural Evaluation: Fixes Dijkstra Blindness
        # Rewards the bot for actively interacting with the board structures
        for (i, j), piece in env.items():
            if piece.get_type() == self.piece_type:
                # Reward our bridges (virtual connections)
                bridges = sum(1 for bi, bj in self._bridges(i, j) if env.get((bi, bj)) and env[(bi, bj)].get_type() == self.piece_type)
                score += bridges * 0.5
                
                # Reward blocking opponent paths
                opp_adj = sum(1 for ni, nj in self._neighbors(i, j) if env.get((ni, nj)) and env[(ni, nj)].get_type() == opp)
                score += opp_adj * 1.5
                
                # Penalize blobbing
                own_adj = sum(1 for ni, nj in self._neighbors(i, j) if env.get((ni, nj)) and env[(ni, nj)].get_type() == self.piece_type)
                score -= own_adj * 0.2
            else:
                # Penalize opponent's bridges
                opp_bridges = sum(1 for bi, bj in self._bridges(i, j) if env.get((bi, bj)) and env[(bi, bj)].get_type() == opp)
                score -= opp_bridges * 0.5

        return score

    def _dijkstra_path(self, env, p_type):
        rows, cols = self.board_dim

        if p_type == 'R':                          # Red connects TOP → BOTTOM (rows)
            starts    = [(0, j) for j in range(cols)]
            goal_test = lambda i, j: i == rows - 1
        else:                                      # Blue connects LEFT → RIGHT (cols)
            starts    = [(i, 0) for i in range(rows)]
            goal_test = lambda i, j: j == cols - 1

        dist = {}
        parent = {}
        heap = []

        for node in starts:
            piece = env.get(node)
            if piece is not None and piece.get_type() != p_type:
                continue
            
            cost = 0 if (piece is not None) else 1
            dist[node] = cost
            parent[node] = None
            heapq.heappush(heap, (cost, node[0], node[1]))

        best_cost = float('inf')
        best_end = None

        while heap:
            cost, i, j = heapq.heappop(heap)
            
            if cost > dist.get((i, j), float('inf')):
                continue
                
            if goal_test(i, j):
                if cost < best_cost:
                    best_cost = cost
                    best_end = (i, j)
                break
                
            for ni, nj in self._neighbors(i, j):
                piece = env.get((ni, nj))
                if piece is not None and piece.get_type() != p_type:
                    continue
                    
                step = 0 if (piece is not None) else 1
                new_cost = cost + step
                
                if new_cost < dist.get((ni, nj), float('inf')):
                    dist[(ni, nj)] = new_cost
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(heap, (new_cost, ni, nj))

        if best_end is None:
            return 999, []

        path_cells = []
        curr = best_end
        while curr is not None:
            piece = env.get(curr)
            if piece is None:
                path_cells.append(curr)
            curr = parent[curr]

        return best_cost, path_cells

    def _neighbors(self, i, j):
        rows, cols = self.board_dim
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                yield ni, nj

    def _bridges(self, i, j):
        rows, cols = self.board_dim
        for di, dj in ((-1, -1), (-2, 1), (-1, 2), (1, 1), (2, -1), (1, -2)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                yield ni, nj