# # 우선순위 큐
import heapq
# heap = [7, 2, 5, 3, 4, 6]
# print(heap)
# heapq.heapify(heap)
# heapq.heappush(heap, 1)
#
# while heap:
#     print(heapq.heappop(heap), end=' ')
# print()

# 최대힙은 ???
temp = [7, 2, 5, 3, 4, 6]
heap2 = []
for i in range(len(temp)):
    heapq.heappush(heap2, -temp[i])
print(heap2)
while heap2:
    print(heapq.heappop(heap2) * -1, end=" ")