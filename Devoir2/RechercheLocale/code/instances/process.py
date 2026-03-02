with open("horaire_D_645_13979.txt", "r") as f:
    lines = f.readlines()

s = set()
for line in lines[2:]:
    part = line[:-1].split(" ")
    s.add(part[0])
    s.add(part[1])

print(len(s))