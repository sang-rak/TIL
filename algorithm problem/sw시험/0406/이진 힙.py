import sys
sys.stdin = open('이진 힙.txt', 'r')

def heapPush(value):
    global heapCount
    heapCount += 1
    heap[heapCount] = value
    current = heapCount   # 마지막 노드가 현재 노드
    parent = current//2   # 부모 노드

    # 부모 노드가 있고 자식보다 더 작을 때
    while parent and heap[parent] > heap[current]:
        heap[parent], heap[current] = heap[current], heap[parent]

        # 이동한 후 갱신
        current = parent
        parent = current // 2

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    temp = list(map(int, input().split()))
    heap = [0] * (N+1)

    heapCount = 0  # 노드가 없는 상태
    for i in range(N):   # 힙에 저장
        heapPush(temp[i])

    parent_sum = 0
    while N >= 1:
        N = N // 2
        parent_sum += heap[N]

    print('#{} {}'.format(tc, parent_sum))