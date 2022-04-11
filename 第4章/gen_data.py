import random
fp = open("./chengji.txt", "w", encoding="utf-8")
def calc_label(x1, x2, x3):
    w1 = 0.72
    w2 = 0.20
    w3 = 0.08
    n1 = x1 * w1
    n2 = x2 * w2
    n3 = x3 * w3
    y = n1 + n2 + n3
    # print(y)
    label = 0
    if y > 80:
        label = 1
    else:
        label = 0
    return label

for i in range(1000):
    x1 = random.randint(30, 100)
    x2 = random.randint(30, 100)
    x3 = random.randint(30, 100)
    label = calc_label(x1, x2, x3)
    # cnt[label] += 1
    print("{0} {1} {2} {3}".format(x1, x2, x3, label))
    fp.write("{0} {1} {2} {3}\n".format(x1, x2, x3, label))
fp.close()
# print("ok,", cnt)
