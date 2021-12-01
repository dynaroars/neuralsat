def dimacs_parse(filename):
    formula = []
    for line in open(filename):
        if line.startswith('c'): 
            continue
        if line.startswith('p'):
            nvars, nclauses = line.split()[2:4]
            continue
        clause = [int(x) for x in line[:-2].split()]
        formula.append(clause)
    return formula, int(nvars)


if __name__ == '__main__':
    print(dimacs_parse('dimacs_cnf.txt'))