import argparse
from queue import PriorityQueue

def parse_state_space_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if not line.strip().startswith('#') and line.strip()]
        initial_state = lines[0]
        goal_states = lines[1].split()
        transitions = {}
        for line in lines[2:]:
            parts = line.split(':')
            state = parts[0].strip()
            transitions[state] = {}
            for transition in parts[1].split():
                next_state, cost = transition.split(',')
                transitions[state][next_state] = float(cost)
        return initial_state, goal_states, transitions
        # initial state -> jedno ime 
        # goal state -> lista imena 
        # transitions -> rjecnik rjecnika -> za jedno ime dobiju se ostala imena i prikladna cijena 

def parse_heuristic_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if not line.strip().startswith('#') and line.strip()]
        heuristics = {}
        for line in lines:
            state, value = line.split(':')
            heuristics[state.strip()] = float(value.strip())
        return heuristics

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def path(self):
        """Rekurzivno rekonstruira put od početnog do trenutnog čvora."""
        if self.parent is None:  # Ako je ovo početni čvor
            return [self]
        else:
            return self.parent.path() + [self]
        
    def __lt__(self, other):
        # Prvo uspoređujemo po cijeni
        if self.cost + self.heuristic < other.cost + other.heuristic: # heuristic je 0 za sve osim A* 
            return True
        elif self.cost == other.cost:
            # Ako su cijene jednake, uspoređujemo abecedno po imenu stanja
            return self.state < other.state
        else:
            return False

class SearchAlgorithm:
    def __init__(self, initial_state, goal_states, transitions, heuristics=None):
        self.initial_node = Node(initial_state)
        self.goal_states = goal_states
        self.transitions = transitions
        self.heuristics = heuristics
        self.closed = set()
        self.path = []
        self.total_cost = 0
        self.final_node = None

    def bfs(self):
        """ Implementacija BFS algoritma """
        open = [self.initial_node] # Ponaša se kao queue, tu čuvamo čitave nodeove, nije atribut klase jer je nekad queue, nekad priority queue, da imamo DFS bilo bi stack
        while open: 
            n = open.pop(0)
            self.closed.add(n.state)
            if n.state in self.goal_states: 
                self.final_node = n 
                self.total_cost = n.cost
                return True 
            for state, transition_cost in sorted(self.transitions[n.state].items(), key=lambda x: x[0]):
                if state not in self.closed: 
                    new_node = Node(state, n, n.cost + transition_cost)
                    open.append(new_node)     
        return False
        

    def ucs(self):
        """ Implementacija UCS algoritma """
        open = PriorityQueue()
        open.put(self.initial_node)
        while open: 
            n = open.get()
            self.closed.add(n.state)
            if n.state in self.goal_states: 
                self.final_node = n 
                self.total_cost = n.cost
                return True 
            for state, transition_cost in self.transitions[n.state].items(): 
                if state not in self.closed: 
                    new_node = Node(state, n, n.cost + transition_cost)
                    open.put(new_node) # Izbjegli smo tuple i dupliranje varijabli kroz dekorator __lt__ jer A* možemo na isti način sortirati samo će mu cost biti g + h
        return False
        

    def a_star(self):
        """ Implementacija algoritma A* """

        # U closedu moramo pamtiti cost pa ćemo tu stavljati čitav node, isto tako ne smije biti set jer trebamo mijenjati kroz iteraciju
        self.closed = []
        open = PriorityQueue()
        self.initial_node.heuristic = self.heuristics[self.initial_node.state]
        open.put(self.initial_node)

        while open: 
            n = open.get()
            self.closed.append(n)
            if n.state in self.goal_states: 
                self.final_node = n 
                self.total_cost = n.cost
                return True 
            for state, transition_cost in self.transitions[n.state].items(): 
                # OVO NE BI BILO POTREBNO AKO JE HEURISTIKA KONZISTENTNA
                flag = False
                # Imamo li taj state u closedu 
                for i in range(len(self.closed)): 
                    if self.closed[i].state == state: 
                        flag = True
                        # Je li jeftiniji sad ili onda
                        if self.closed[i].cost <= n.cost + transition_cost: 
                            break # Nikad neće isto stanje biti dva puta u closedu
                        else: 
                            self.closed.pop(i)
                            new_node = Node(state, n, n.cost + transition_cost, self.heuristics[state])
                            open.put(new_node)
                            break
                # Imamo li taj state u openu
                if not flag:
                    for i in range(open.qsize()): 
                        if open.queue[i].state == state: 
                            flag = True
                            # Je li jeftiniji sad ili onda 
                            if open.queue[i].cost <= n.cost + transition_cost: 
                                break
                            else: 
                                open.queue.pop(i)
                                new_node = Node(state, n, n.cost + transition_cost, self.heuristics[state])
                                open.put(new_node)
                                break
                if not flag:
                    new_node = Node(state, n, n.cost + transition_cost, self.heuristics[state])
                    open.put(new_node)
        return False

    def __str__(self):
        if self.final_node:  # Ako postoji put
            self.path = self.final_node.path()
            path_str = " => ".join([node.state for node in self.path])
            return f"[FOUND_SOLUTION]: yes\n" \
                f"[STATES_VISITED]: {len(self.closed)}\n" \
                f"[PATH_LENGTH]: {len(self.path)}\n" \
                f"[TOTAL_COST]: {self.total_cost:.1f}\n" \
                f"[PATH]: {path_str}"
        else:  # Ako put ne postoji
                return f"[FOUND_SOLUTION]: no"

class HeuristicChecker:
    def __init__(self, heuristics, goal_states, transitions):
        self.heuristics = heuristics
        self.goal_states = goal_states
        self.transitions = transitions

    def calculate_hStar(self, state):
        # Htio sam dinamički pristupiti zbog memoizacije ali za vece skupove je dubina rekurzije prevelika
        # Pokusati cu alternativno iskoristiti UCS metodu iz klase SearchAlgorithm makar nije bas modularna jer nisam imao u planu ovo 
        ucs_algo = SearchAlgorithm(state, self.goal_states, self.transitions)
        ucs_algo.ucs()
        return float(ucs_algo.total_cost)

    def check_optimistic(self):
        optimistic = ""
        for state, h in sorted(self.heuristics.items(), key=lambda x: x[0]):
            hStar = self.calculate_hStar(state)
            if h > hStar: 
                optimistic = "not"
                currOptimistic = "[ERR]"
            else: 
                currOptimistic = "[OK]"
            print(f"[CONDITION]: {currOptimistic} h({state}) <= h*: {h:.1f} <= {hStar:.1f}")

        print(f"[CONCLUSION]: Heuristic is {optimistic} optimistic.")

        return 
    
    def check_consistent(self):
        consistent = ""
        for state, transitions in self.transitions.items():
            for adjacent_state, adjacent_state_cost in transitions.items(): 
                if self.heuristics[state] <= self.heuristics[adjacent_state] + adjacent_state_cost: 
                    currConsistent = "[OK]"
                else: 
                    currConsistent = "[ERR]"
                    consistent = "not"
                print(f"[CONDITION]: {currConsistent} h({state}) <= h({adjacent_state}) + c: {self.heuristics[state]:.1f} <= {self.heuristics[adjacent_state]:.1f} + {adjacent_state_cost:.1f}")
        print(f"[CONCLUSION]: Heuristic is {consistent} consistent.")


def main():
    # Inicijalizacija parsera argumenata
    parser = argparse.ArgumentParser(description="Search Algorithm Implementation")
    parser.add_argument('--alg', type=str, choices=['bfs', 'ucs', 'astar'], help='Algorithm for search (bfs, ucs, or astar)')
    parser.add_argument('--ss', type=str, help='Path to the state space descriptor')
    parser.add_argument('--h', type=str, help='Path to the heuristic descriptor')
    parser.add_argument('--check-optimistic', action='store_true', help='Flag to check if the heuristic is optimistic')
    parser.add_argument('--check-consistent', action='store_true', help='Flag to check if the heuristic is consistent')

    # Parsiranje argumenata
    args = parser.parse_args()
    if args.ss:
        initial_state, goal_states, transitions = parse_state_space_file(args.ss)
    
    search_algorithm = SearchAlgorithm(initial_state, goal_states, transitions)
    
    # Provjeri koji je algoritam odabran i pokreni ga
    if args.alg in ['bfs', 'ucs', 'astar']:
        if args.alg == 'bfs':
            print(f"# BFS {args.ss}")
            search_algorithm.bfs()
        elif args.alg == 'ucs':
            print(f"# UCS {args.ss}")
            search_algorithm.ucs()
        elif args.alg == 'astar':
            print(f"# A-STAR {args.ss}")
            search_algorithm.heuristics = parse_heuristic_file(args.h)
            search_algorithm.a_star()
        print(search_algorithm)
    if args.check_optimistic:
        print(f"# HEURISTIC-OPTIMISTIC {args.h}")
        heuristic_checker = HeuristicChecker(parse_heuristic_file(args.h), goal_states, transitions)
        heuristic_checker.check_optimistic()
    if args.check_consistent:
        print(f"# HEURISTIC-CONSISTENT {args.h}")
        heuristic_checker = HeuristicChecker(parse_heuristic_file(args.h), goal_states, transitions)
        heuristic_checker.check_consistent()

if __name__ == "__main__":
    main()

