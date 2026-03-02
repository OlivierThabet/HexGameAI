import math
from solver_advanced import solve
from schedule import Schedule
print("***********************************************************")
print("[INFO] Start the autograding: train network design")
print("***********************************************************\n\n")

instances = ["horaire_A_11_20.txt","horaire_B_23_71.txt", "horaire_C_121_3960.txt", "horaire_D_558_13979.txt"]
scores_randoms = [9, 15, 84, 354] 
scores_secrets = [5, 6, 25, 40]
instances = [Schedule("instances/"+i) for i in instances]
scores_beaten = [[],[]]
has_failed = False
for i in range(len(instances)):
    print("***********************************************************")
    print("[INFO] autograding: instance",instances[i].instance_name)
    try:
        solution = solve(instances[i])
        score = instances[i].get_n_creneaux(solution)
        valid = instances[i].verify_solution(solution)
        if not valid:
            raise Exception("Invalid solution")
        print("[INFO] RUN: passed")
    except Exception as e:
        print("[INFO] RUN: failed :",e)
        score = math.inf
        has_failed = True
        
    print("[INFO] score: ",score)
    print("[INFO] Random player beaten ("+str(scores_randoms[i])+"):", score < scores_randoms[i])
    print("[INFO] Secret player beaten ("+str(scores_secrets[i])+"):", score < scores_secrets[i])
    scores_beaten[0].append(score < scores_randoms[i])
    scores_beaten[1].append(score < scores_secrets[i])
    print("***********************************************************\n\n")

print("***********************************************************")
print("[INFO] autograding: summary")
if has_failed:
    print("[INFO] RUN: failed, 0/10")
else:
    
    print("[INFO] RUN: passed, >0/10")
    if sum(scores_beaten[0]) < 4:
        print("[INFO] Random player beaten: failed, <5/10")
        print("[INFO] Hint: vérifiez que votre recherche locale cherche à minimiser le coût, vérifiez que la fonction de voisinage est correcte, que la recherche locale s'arrête bien quand il n'y a plus d'amélioration possible")
    else:
        print("[INFO] Random player beaten: passed, >=5/10")
        if (sum(scores_beaten[1]) < 4):
            print("[INFO] Secret player beaten: failed, <8/10")
            print("[INFO] Hint: Utilisez des metaheuristiques comme le recuit simulé ou le restart")
        else:
            print("[INFO] Secret player beaten: passed, >=8/10")
print("***********************************************************\n\n")
    
