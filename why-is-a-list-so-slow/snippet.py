import random, time

tickets = [f"P{i:07d}" for i in range(10**6)]
seats = dict.fromkeys(tickets)
checks = random.sample(tickets, 1000)

t = time.perf_counter()
sum(c in tickets for c in checks)
print(round(time.perf_counter() - t, 4))

t = time.perf_counter()
sum(c in seats for c in checks)
print(round(time.perf_counter() - t, 4))
