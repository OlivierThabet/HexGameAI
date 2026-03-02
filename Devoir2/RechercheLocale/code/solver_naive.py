from schedule import Schedule

def solve(schedule: Schedule):
    """
    Naive way of solving the problem: assigning a different time slot to each course.
    :param schedule: object describing the input
    :return: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    """

    solution = dict()

    time_slot_idx = 1
    for c in schedule.course_list:

        assignation = time_slot_idx
        solution[c] = assignation
        time_slot_idx += 1
        
    return solution