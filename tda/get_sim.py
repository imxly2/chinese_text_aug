import pickle

stop = open('stop.txt', encoding='utf8').read().split('\n')
d = {}
for line in open('syn.txt'):
    line = line.strip().split()[1:]
    line = [x for x in line if len(x) > 1 and x not in stop]

    if len(line) == 1:
        continue
    for i, v in enumerate(line):
        # if len(v) == 1:
        #     continue
        d[v] = [x for x in line if x != v]

print(d)
pickle.dump(d, open('syn.pkl', 'wb+'))
