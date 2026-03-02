import argparse
import time
from schedule import Schedule
import solver_naive
import solver_advanced

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--infile', type=str, default='input')
    parser.add_argument('--outfile', type=str, default='solution.txt')
    parser.add_argument('--visufile', type=str, default='visualization.png')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    schedule = Schedule(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: Academic schedule")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visufile)
    print("[INFO] number of courses: %s" % (schedule.conflict_graph.number_of_nodes()))
    print("[INFO] number of conflicts: %s" % (schedule.conflict_graph.number_of_edges()))
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(schedule)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(schedule)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60,2)

    schedule.display_solution(solution,args.visufile)
    schedule.save_solution(solution, args.outfile)


    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Number of time slots required : %s" % schedule.get_n_creneaux(solution))
    print("[INFO] Sanity check passed : %s" % schedule.verify_solution(solution))
    print("***********************************************************")