import matplotlib.pyplot as plt
import networkx as nx
import random as r

class Schedule():

    def __init__(self,filename):
        """Initialize the schedule structure
        :param filename: path to the file containing the data"""
        self.instance_name = filename
        self.conflict_graph = nx.Graph()            # Conflicts are modeled using a graph

        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines[2:]:
            l = line[:-1].split()
            i,j = l[0],l[1]
            self.conflict_graph.add_edge(i,j)

        self.course_list = self.conflict_graph.nodes        # Each node represents a course
        self.conflict_list = self.conflict_graph.edges      # Each edge represents a conflict between two courses

    def get_node_conflicts(self, node):
        """Get all conflicts of a given node
        :param node: the node representing a course
        :return: a set containing all conflicts of the given node"""
        return set(self.conflict_graph.neighbors(node))
        
    def get_n_creneaux(self,solution):
        """Calculates the number of time slots used by a solution
        :param solution: a dictionnary where the keys are the list of the courses and the values are the time periods associated
        :return: an int containing the number of used time slots"""
        return len(set(solution.values()))

    def verify_solution(self,solution):
        """Verify if a solution is feasible or not
        :param solution: a dictionnary where the keys are the list of the courses and the values are the time periods associated"""
        assert sum(solution[a[0]] == solution[a[1]] for a in self.conflict_list) == 0, "Solution invalide"
        return True
    
    def save_solution(self, solution, filename):
        """Writes a given solution
        :param solution: a dictionnary where the keys are the list of the courses and the values are the time periods associated
        :param filename: the path to the file where the solution will be written"""
        with open(filename,'w') as f:
            f.write("%s\n%s\n%s\n" % (self.conflict_graph.number_of_nodes(), self.conflict_graph.number_of_edges(), self.get_n_creneaux(solution)))
            for i in solution:
                f.write("%s %s\n" % (i, solution[i]))

    def display_solution(self, solution=[], filename="out.png"):
        """Draw a given solution
        :param solution: a dictionnary where the keys are the list of the courses and the values are the time periods associated
        :param filename: the path to the file where the solution will be drawn"""
        colors=dict()
        pos = nx.spring_layout(self.conflict_graph, seed=10)
        nx.draw_networkx_edges(self.conflict_graph, pos)
        for i in solution:
            if solution[i] not in colors:
                colors[solution[i]]=(r.randint(100,255)/255,r.randint(100,255)/255,r.randint(100,255)/255)
            plt.text(pos[i][0]-0.125, pos[i][1]-0.05, i, bbox=dict(facecolor=colors[solution[i]],edgecolor='black', boxstyle='round,pad=0.2'))
        
        plt.xlim(min([pos[i][0] for i in solution]), max([pos[i][0] for i in solution]))
        plt.ylim(min([pos[i][1] for i in solution]), max([pos[i][1] for i in solution]))
        plt.axis('off')
        plt.savefig(filename)
