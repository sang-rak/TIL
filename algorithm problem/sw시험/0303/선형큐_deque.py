import collections
deq = collections.deque([1])
print(deq)
deq.append(1)  # enQueue
deq.append(1)
deq.append(1)

print(deq.popleft())  # deQueue
print(deq.popleft())
print(deq.popleft())