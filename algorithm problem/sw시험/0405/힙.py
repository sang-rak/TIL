def heapPush(value):
    global heapCount
    heapCount += 1           # 마지막 노드 번호 증가
    heap[heapCount] = value  # 마지막 노드에 값 저장
    current = heapCount      # 마지막 노드가 현재노드
    parent = current // 2    # 부모노드 계산

    # 루트가 아니고, 부모 노드의 값이 자식노드의 값보다 더 크면 -> swap
    while parent and heap[parent] > heap[current]:
        heap[parent], heap[current] = heap[current], heap[parent]

        current = parent
        parent = current // 2

def heapPop():
    global heapCount           # 리턴값
    retValue = heap[1]
    heap[1] = heap[heapCount]  # 마지막 노드를 루트로 이동
    heap[heapCount] = 0        # 마지막 노드 지움
    heapCount -= 1             # 힙 카운트 감소

    parent = 1
    child = 2 * parent

    # 오른쪽 자식이 존재하면 왼쪽과 비교해서 작은값을 child
    if child + 1 <= heapCount:
        # 더 큰값을 child 로
        if heap[child] > heap[child+1]:
            child = child + 1

    # 자식노드가 힙카운트보다 작고, 부모노드가 자식보다 크면 swip
    while child <= heapCount and heap[parent] > heap[child]:
        heap[parent], heap[child] = heap[child], heap[parent]

        parent = child
        child = 2 * parent

        # 오른쪽 자식이 존재하면 왼쪽과 비교해서 작은값을 child
        if child + 1 <= heapCount:
            # 더 큰값을 child 로
            if heap[child] > heap[child + 1]:
                child = child + 1

    return retValue


# 최소힙
heapCount = 0  # 노드가 하나도 없는 상태
temp = [7, 2, 5, 3, 4, 6]
N = len(temp)
heap = [0] * (N+1)

for i in range(N):  # 힙에 저장
    heapPush(temp[i])

# for i in range(N): # 출력
#     heapPop(temp[1])
print(heap)