from sets import Set
from random import choice


def chance_of_visiting_all_states(iterations, k, n):
    r = range(n)
    total = 0
    for i in range(iterations):
        s = Set()
        for j in range(k):
            s.add(choice(r))
            if len(s) == n:
                total +=1
                break
    return float(total)/iterations

steps = 10000
print "Chance of visiting all states in {st} steps: {ch}".format \
    (st=steps, ch=chance_of_visiting_all_states(20000, steps, 1536))