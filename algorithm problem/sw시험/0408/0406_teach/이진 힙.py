import sys
sys.stdin = open('이진 힙.txt', 'r')

def heapPush(value):
    global heapCount
    heapCount += 1           # 마지막 노드 번호 증가
    heap[heapCount] = value  # 마지막 노드에 값 저장
    child = heapCount        # 마지막 노드가 child
    parent = child // 2      # 머지막 노드의 parent

    # 루트가 아니고, 부모노드 > 자식노드 -> swap
    while parent and heap[parent] > heap[child]:
        heap[parent], heap[child] = heap[child], heap[parent]
        child = parent
        parent = child // 2

def heapPop():
    global heapCount
    retValue = heap[1]         # 루트값 리턴
    heap[1] = heap[heapCount]  # 마지막 노드를 루트로 이동
    heap[heapCount] = 0        # 마지막 노드 지우기
    heapCount -= 1             # 카운트 감소
    parent = 1
    child = 2 * parent         # 왼쪽 자식

    if child + 1 <= heapCount: # 오른쪽 자식 존재 유무 확인
        if heap[child] > heap[child+1]: # 오른쪽 자식이 작으면
           child = child + 1

    # 자식이 존재하고 부모 > 자식 -> swap
    while child <= heapCount and heap[parent] > heap[child]:
        heap[parent], heap[child] = heap[child], heap[parent]
        parent = child
        child = 2 * parent  # 왼쪽 자식

        if child + 1 <= heapCount:  # 오른쪽 자식 존재 유무 확인
            if heap[child] > heap[child + 1]:  # 오른쪽 자식이 작으면
                child = child + 1

    return retValue

def find():
    parent = heapCount // 2
    sum = 0
    while parent:
        sum += heap[parent]
        parent = parent // 2

    return sum

T = int(input())

for tc in range(1, T+1):
    N = int(input())    # 정점의 수
    heap = [0] * (N+1)  # 완전이진트리 1차원 리스트
    heapCount = 0
    temp = list(map(int, input().split()))

    for i in range(N):  # 힙에 저장
        heapPush(temp[i])

    # 확인용 출력
    # for i in range(N):     # 출력
    #     print(heapPop(), end=' ')


    print("#{} {}".format(tc, find()))
