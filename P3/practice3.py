t = ["ABCD", "ACDF", "ACDEG", "ABDF", "BCG", "DFG", "ABG", "CDFG"]

min_support = 3

base = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

my_set = [[]]

for i in base:
    count = 0
    for j in t:
        if j.count(i) > 0:
            count = count + 1
    # print("count" + i + " = " + str(count))
    if count > min_support:
        my_set[0].append(set(i))

k = 0
while True:
    new_set = set()
    for c in base:
        for i in my_set[k]:
            tmp = i.copy()
            tmp.add(c)
            if i == tmp:
                continue
            count = 0
            for j in t:
                nope = 0
                for x in i.add(c):
                    if j.count(x) == 0:
                        nope = 1
                        break
                    count = count + 1 if nope == 0 else count
            if count > min_support:
                new_set.add((i + c))
            # print("count" + i + c + " = " + str(count))
    if len(new_set) == 0:
        break
    else:
        my_set.append(new_set)
        print(new_set)
        k = k + 1
print()
