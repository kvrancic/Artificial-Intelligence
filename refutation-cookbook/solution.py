import argparse

class Clause:
    def __init__(self, literals, id, parents=None, printid=None):
        self.literals = frozenset(literals) # literali klauzule
        self.id = id  # redni broj klauzule
        self.parents = sorted(parents) if parents else None # identifikatori roditeljskih klauzula
        self.printid = printid # redni broj klauzule za ispis

        
    def __eq__(self, other):
        # služi za provjeru imamo li već istu takvu klauzulu (makar to provjerava i redundancija tako da je ova metoda *reduntantna*)
        return set(self.literals) == set(other.literals)
    
    def __str__(self):
        # vraća string reprezentaciju klauzule
        literals_str = ' v '.join(self.literals)
        if self.parents:
            if not self.literals:
                return f"{self.id}. NIL ({', '.join(map(str, self.parents))})"
            else:
                return f"{self.id}. {literals_str} ({', '.join(map(str, self.parents))})"
        else:
            if not self.literals:
                return f"{self.id}. NIL"
            else:
                return f"{self.id}. {literals_str}"
    
    def __hash__(self):
        return hash((self.literals, self.id))

    def __lt__(self, other):
        return self.id < other.id

def parse_clauses(file_path, isCooking=False):
    clausesLiterals = []
    #print("\n=== Input clauses ===")
    with open(file_path, 'r') as file:
        lines = [line.lower().strip() for line in file if not line.strip().startswith('#') and line.strip()]
        #print(lines)
        for line in lines:
            clausesLiterals.append(line.split(' v ')) # clauses je sada lista listi: [[a, b], [c, d], ...] => posto smo u CNF elementi podliste su povezani s 'or', dok su medusobno listep povezane s 'and'
    
    # ako u startu nema goal klauzule -> kasnije stvaram objekte
    if isCooking:
        #clausesLiterals = [set(clause) for clause in clausesLiterals]
        return clausesLiterals 
    
    else:
        goal_clause = clausesLiterals.pop() # zadnji element je goal clause

        len_input_clauses = len(clausesLiterals)

        clausesLiterals = add_negated_goal_clause(clausesLiterals, goal_clause) # dodajemo negiranu goal klauzulu u skup klauzula
        
        # pretvori svaku klauzulu u set kako bi se faktorizirale 
        clausesLiterals = [set(clause) for clause in clausesLiterals]
        # pretvori setove u klase
        clauses = [Clause(list(clause), id) for id, clause in enumerate(clausesLiterals, 1)]

        return clauses, clauses[len_input_clauses:], goal_clause
        

def add_negated_goal_clause(clauses, goal_clause):
    for literal in goal_clause:
        # bitno je da dodajemo svaki literal zasebno jer ~(a v b) = ~a ^ ~b (De Morgan)
        clauses.append([f'~{literal}']) if literal[0] != '~' else clauses.append([literal[1:]]) # ako je literal negiran, makni negaciju, inace dodaj negaciju
        
    return clauses

def resolution(clauses, negated_goal_clauses):
    # clauses is a list of Clause objects, negated_clauses is a list of negated goal clause literals 
    new_clauses = []
    all_clauses = clauses # treba nam za backtracking ispis, a zbog redundancije ce neke nestati iz clauses 
    sos = negated_goal_clauses
    id_counter = len(clauses) + 1
    sos_to_add = []
    while True: 
        clauses = [clause for clause in clauses if not is_redundant(clause, clauses) and not is_tautology(clause.literals)]
        sos = [clause for clause in sos if not is_redundant(clause, sos) and not is_tautology(clause.literals)]
        """ print("Cycle starting...")
        print("Clauses:")
        for clause in clauses:
            print(clause)
        print("SOS: ")
        for clause in sos:
            print(clause) """
        # iterate through literals for each clause in clauses
        for ci in reversed(clauses): # reversed to prioritize new clauses
            for cj in reversed(sos): 
                #print(f"Resolving {ci} and {cj}")
                resolved, resolvent = resolve_pair(ci, cj, id_counter)
                if resolved:
                    #print(f"Resolved: {resolvent}")
                    if resolvent in clauses or resolvent in new_clauses: 
                        continue # izbjegni duplikate 
                    if not resolvent.literals: # prazna klauzula - GOTOVO 
                        backtrack_print(resolvent, all_clauses + new_clauses)
                        return True
                    # treba vidjeti je li brze propustiti koji redudantni ovako ili stalno sve provjeravati
                    if is_redundant(resolvent, clauses + new_clauses):
                        continue # izbjegni redundante klauzule (klauzule podskup druge klauzule)
                    #print(f"Passed: {resolvent}")
                    id_counter += 1
                    new_clauses.append(resolvent)
        if not new_clauses:
            return False
        clauses += new_clauses
        all_clauses += new_clauses
        sos += new_clauses
        new_clauses = []

def resolve_pair(ci, cj, id_counter):
    for literal in ci.literals:
        if negate_literal(literal) in cj.literals:
            # Kreiramo novu klauzulu koja isključuje suprotstavljene literale
            new_clause_literals = (set(ci.literals) | set(cj.literals)) - {literal, negate_literal(literal)}
            # Provjera da li je nova klauzula tautologija
            if not is_tautology(new_clause_literals):
                resolvent = Clause(list(new_clause_literals), id_counter, parents=[ci.id, cj.id])
                return True, resolvent
            else: 
                return False, None
    return False, None


def negate_literal(literal):
    return literal[1:] if literal.startswith('~') else f'~{literal}'

def is_tautology(literals):
    for literal in literals:
        if negate_literal(literal) in literals:
            return True
    return False

def is_redundant(clause, clauses):
    for existing_clause in clauses:
        if set(existing_clause.literals).issubset(set(clause.literals)) and len(clause.literals) != len(existing_clause.literals): # da ne izbaci sama sebe
            return True  # Klauzula je redundantna ako je nadskup neke druge klauzule (sve klauzule su povezane s "and")
    return False

# ovdje sam se slomio da bi ID-jevi išli od 1 itd., a ne da budu originalni ID-jevi klauzula
def backtrack_print(resolvent, clauses):
    # Prvo sakupljamo sve relevantne klauzule u jednu listu za lakše rukovanje
    relevant_clauses = collect_relevant_clauses(resolvent, clauses)

    # Zatim sortiramo te klauzule po njihovom ID-u
    sorted_clauses = sorted(relevant_clauses, key=lambda c: c.id)

    # Pronalazimo indeks prve klauzule koja ima roditelje
    first_with_parents = next((i for i, c in enumerate(sorted_clauses) if c.parents), len(sorted_clauses))

    # Ispisujemo klauzule bez roditelja
    for i, clause in enumerate(sorted_clauses[:first_with_parents], start=1):
        literals_str = 'NIL' if not clause.literals else ' v '.join(clause.literals)
        print(f"{i}. {literals_str}")
        clause.printid = i

    # Ispisujemo separator ako postoji barem jedna klauzula s roditeljima
    if first_with_parents < len(sorted_clauses):
        print("=" * 30)

    # Ispisujemo klauzule s roditeljima
    for i, clause in enumerate(sorted_clauses[first_with_parents:], start=first_with_parents + 1):
        literals_str = 'NIL' if not clause.literals else ' v '.join(clause.literals)
        parents_str = ', '.join(str(c.printid) for c in sorted_clauses if c.id in clause.parents)
        print(f"{i}. {literals_str} ({parents_str})")
        clause.printid = i

    print("=" * 30)
    return

def collect_relevant_clauses(resolvent, clauses):
    # Rekurzivno sakupljanje svih klauzula koje su dovele do resolventa
    relevant_clauses = set()

    def collect(clause):
        if clause not in relevant_clauses:
            relevant_clauses.add(clause)
            for parent_id in clause.parents or []:
                parent = next((c for c in clauses if c.id == parent_id), None)
                if parent:
                    collect(parent)

    collect(resolvent)
    return list(relevant_clauses)

def execute_command(oldClausesStrings, command):
    # Obrada i izvršavanje pojedinačne korisničke naredbe
    inputString, action = command[:-2], command[-1]
    clausesStrings = oldClausesStrings.copy()
    if action == '?':
        # Provjeri valjanost klauzule
        len_input_clauses = len(clausesStrings)
        goal_clause = inputString.split(' v ')
        clausesStrings = add_negated_goal_clause(clausesStrings, goal_clause)
        clausesStrings = [set(clause) for clause in clausesStrings]
        clauses = [Clause(list(clause), id) for id, clause in enumerate(clausesStrings, 1)]
        
        if resolution(clauses, clauses[len_input_clauses:]):
            print("[CONCLUSION]:", inputString, "is true")
        else:
            print("[CONCLUSION]:", inputString, "is unknown")

        return oldClausesStrings # ne mijenjamo bazu znanja

    elif action == '+':
        # Dodaj klauzulu u bazu znanja
        clausesStrings.append(inputString.split(' v '))
        print("Added", inputString)
        return clausesStrings
    elif action == '-':
        # Izbriši klauzulu iz baze znanja ako postoji
        #print("Trying to remove ", inputString)
        #print("Old clauses: ", clausesStrings)

        for clause in clausesStrings:
            if clause == inputString.split(' v '):
                clausesStrings.remove(clause)
                print("Removed", inputString)
                #print("New clauses: ", clausesStrings)
                return clausesStrings
        print("Clause not found in the list, nothing was removed")
        return clausesStrings
            

def process_commands(clausesStrings, commands_file):
    # Učitava i izvršava korisničke naredbe iz datoteke
    with open(commands_file, 'r') as file:
        for line in file:
            command = line.lower().strip()
            print("\n")
            print("User's command:", command)
            clausesStrings = execute_command(clausesStrings, command)


def main(): 
    parser = argparse.ArgumentParser(description='Refutation resolution algorithm')
    parser.add_argument("mode", type=str, choices=['resolution', 'cooking'], help="Operation mode")
    parser.add_argument("clauses_file", type=str, help="File path for the list of clauses")
    parser.add_argument("commands_file", type=str, nargs='?', help="File path for the list of user commands", default="")

    args = parser.parse_args()
    
    if args.mode == "resolution":
        clauses, negated_goal_clauses, goal_clause = parse_clauses(args.clauses_file)
        if resolution(clauses, negated_goal_clauses): 
            print(f"[CONCLUSION]: {' v '.join(goal_clause)} is true")
        else: 
            print(f"[CONCLUSION]: {' v '.join(goal_clause)} is unknown")

    elif args.mode == "cooking":
        print("Constructed with knowledge:")
        clausesStrings = parse_clauses(args.clauses_file, isCooking=True) # ovo je za sad obična lista stringova
        for clause in clausesStrings:
            print(' v '.join(clause))

        if args.commands_file:
            process_commands(clausesStrings, args.commands_file)
        else:
            print("No commands provided")


            #for command in commands:
            #   print(f"{command[0]} {command[1]}")

if __name__ == "__main__":
    main()