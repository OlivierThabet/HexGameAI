#2294559 et 2212749
import random


def build_neighbors(schedule):
    return {course: set(schedule.get_node_conflicts(course)) for course in schedule.course_list}


def greedy_coloring(course_list, neighbors):
    solution = {}
    for course in course_list:
        # Récupère les créneaux déjà utilisés par les voisins en conflit déjà assignés.
        used_slots = {solution[n] for n in neighbors[course] if n in solution}
        # On assigne le plus petit créneau valide.
        slot = 1
        while slot in used_slots:
            slot += 1
        solution[course] = slot
    return solution


def max_slot(solution):
    if solution:
        return max(solution.values()) 
    else:
        return 0


def can_move(course, target_slot, solution, neighbors):
    for neighbor in neighbors[course]:
        if solution[neighbor] == target_slot:
            return False
    return True


def try_reduce_one_slot(solution, neighbors):
    # Cible le plus grand créneau utilisé et tente de le faire disparaître.
    current_max_slot = max_slot(solution)
    if current_max_slot <= 1:
        return False

    moved = True
    while moved:
        moved = False
        for course in list(solution.keys()):
            if solution[course] != current_max_slot:
                continue
            # Essaie de déplacer ce cours vers un créneau plus petit tout en restant valide.
            for trial_slot in range(1, current_max_slot):
                if can_move(course, trial_slot, solution, neighbors):
                    solution[course] = trial_slot
                    moved = True
                    break

    # S'il ne reste plus de cours sur ce créneau, on a réduit le nombre de créneaux.
    if all(solution[course] != current_max_slot for course in solution):
        return True
    return False


def solve(schedule):
    neighbors = build_neighbors(schedule)
    course_list = list(schedule.course_list)
    degrees = {course: len(neighbors[course]) for course in course_list}

    # Point de départ déterministe en triant les cours par degré décroissant.
    initial_order = sorted(course_list, key=lambda course: (-degrees[course], course))
    best_solution = greedy_coloring(initial_order, neighbors)
    best_score = max_slot(best_solution)

    # Recherche locale répétée pour éviter de rester coincé dans un minimum local.
    # On garde le meilleur résultat des 50 redémarrages.
    restart_count = 50
    for _ in range(restart_count):
        random_order = course_list[:]
        # Mélange aléatoire de l'ordre des cours pour générer un nouveau point de départ.
        random.shuffle(random_order)
        current_solution = greedy_coloring(random_order, neighbors)

        # Descente locale basée sur la recherche de réduction du nombre de créneaux utilisés.
        improved = True
        while improved:
            improved = try_reduce_one_slot(current_solution, neighbors)

        current_score = max_slot(current_solution)
        if current_score < best_score:
            best_solution = current_solution.copy()
            best_score = current_score

    return best_solution
