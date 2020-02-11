## --driver-memory 5g (probably more than enough)

def form_pairs(tup):
    l = []
    # for each tup[0] get n1 and n2 that has tup[0] as common friend
    for elem1 in tup[1]:
        for elem2 in tup[1]:
            if elem1 < elem2:
                friend = tup[0]
                l.append(((elem1, elem2), [friend]))
    return l

def check(line):
    if len(line.split()) > 1:
           return len((line.split())[1].split(',')) > 1
    return False

import sys
from pyspark import SparkContext

file = sys.argv[1]
out_file = sys.argv[2]

sc = SparkContext(appName="Common Friends 2")
f = sc.textFile(file)

# Obtaining edges
adlists = f.filter(check).map(lambda line: line.split())
# [... (node, [adjacent nodes...]) ...]
adlists = adlists.map(lambda l: (l[0], l[1].split(',')))

#For every co-occurrence of i and j (> i) we add ((i, j), 1)
pairs = adlists.flatMap(form_pairs)
# [ ((n1, n2), [common friend]) ...]
cf = pairs.reduceByKey(lambda x,y: x + y)
# sum common friend for each equal x, y
# [ ((x, y), [common friends...]) ...]

sim = cf.collect()

with open(out_file, "w") as f:
    for t in sorted(sim):
        s = str(t[0][0]) + "," + str(t[0][1]) + "\t" + "%s" % repr(list(map(int, t[1]))) + "\n"
        f.write(s)



