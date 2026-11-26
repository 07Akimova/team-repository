import matplotlib.pyplot as plt

data = []

with open("data/vova_90_f_160.txt", "r") as f:
    for i in range(1, 5):
        f.readline()

    n = 0
    
    while True:
        line = f.readline()
        if not line:
            break
        
        data.append(int(line))
        n += 1

plt.plot(data)
plt.show()