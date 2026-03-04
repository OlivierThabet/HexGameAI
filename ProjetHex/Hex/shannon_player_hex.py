import numpy as np
import copy
from scipy.sparse.linalg import cg
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_hex import GameStateHex

class MyPlayer(PlayerHex):
    """
    Player class for Hex game using Minimax with Alpha-Beta pruning
    and a True Electrical Current (Amperage) heuristic.
    """

    def __init__(self, piece_type: str, name: str = "AmperesBot"):#
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:

        alpha = float('-inf')
        beta = float('inf')
        depth = 2

        _, action = self.minimax(current_state, depth, alpha, beta, True)

        return current_state.convert_stateful_action_to_stateless_action(action)

    def minimax(self, state: GameStateHex, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        if state.is_done() or depth == 0:
            return self.evaluate_state(state), None
            
        best_action = None

        if maximizingPlayer:
            max_eval = float('-inf')
            for action in state.generate_possible_stateful_actions():
                eval, _ = self.minimax(action.get_next_game_state(), depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in state.generate_possible_stateful_actions():
                eval, _ = self.minimax(action.get_next_game_state(), depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def evaluate_state(self, state: GameStateHex) -> float:
        """ Score = (My Equivalent Conductance) - (Opponent's Equivalent Conductance) """
        opp_piece = 'B' if self.piece_type == 'R' else 'R'
        my_dir = "HAUTBAS" if self.piece_type == 'R' else "GAUCHEDROITE"
        opp_dir = "GAUCHEDROITE" if self.piece_type == 'R' else "HAUTBAS"
        
        my_c = self.calculate_amperage(state, self.piece_type, my_dir)
        opp_c = self.calculate_amperage(state, opp_piece, opp_dir)
        
        return my_c - opp_c

    def calculate_amperage(self, state: GameStateHex, target_piece: str, direction: str) -> float:
        N = state.get_rep().get_dimensions()[0]
        nb_nodes = N * N
        G = np.zeros((nb_nodes, nb_nodes))
        I = np.zeros(nb_nodes)
        
        env = state.get_rep().get_env()
        directions = [(0,-1),(-1,0),(-1,1),(0,1),(1,0),(1,-1)] # Dans l'ordre (ligne, colonne): gauche, gauche-haut, droite-haut, droite, droite-bas, gauche-bas

        # Pre-map node resistances for speed
        R = np.ones((N, N)) # 1 Ohm for empty space
        for (r, c), piece in env.items():
            if piece.get_type() == target_piece:
                R[r, c] = 0.001 # Superconductor (My pieces)
            else:
                R[r, c] = float('inf') # Insulator (Opponent pieces)

        # Build Laplacian Matrix
        for r in range(N):
            for c in range(N):
                R_i = R[r, c]
                if R_i == float('inf'):
                    node = r * N + c
                    G[node, node] = 1.0 # Isolate disconnected nodes
                    continue

                node = r * N + c
                diag_sum = 0.0
                
                # Connect internal neighbors
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        R_j = R[nr, nc]
                        if R_j != float('inf'):
                            n_node = nr * N + nc
                            # Edge Conductance
                            C_ij = 2.0 / (R_i + R_j) 
                            G[node, n_node] -= C_ij
                            diag_sum += C_ij

                # Boundary Connections (Rails)
                C_source = 0.0
                C_sink = 0.0
                
                if direction == "HAUTBAS":
                    if r == 0:     C_source = 2.0 / R_i
                    if r == N - 1: C_sink = 2.0 / R_i
                else: 
                    if c == 0:     C_source = 2.0 / R_i
                    if c == N - 1: C_sink = 2.0 / R_i
                
                G[node, node] = diag_sum + C_source + C_sink
                I[node] = C_source # 1V Source pushes current

        try:
            V, exit_code = cg(G, I, rtol=1e-5)
            
            # THE FIX: Calculate the Total Amperage pulled from the 1V Source rail
            total_current = 0.0
            for i in range(N):
                if direction == "HAUTBAS":
                    r, c = 0, i
                else:
                    r, c = i, 0
                    
                R_i = R[r, c]
                if R_i != float('inf'):
                    node = r * N + c
                    C_source = 2.0 / R_i
                    # Current entering board = Conductance * (Source_Voltage - Node_Voltage)
                    total_current += C_source * (1.0 - V[node])
                    
            return float(total_current)
            
        except np.linalg.LinAlgError:
            return 0.0