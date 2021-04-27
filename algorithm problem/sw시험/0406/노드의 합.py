import sys
sys.stdin = open('노드의 합.txt', 'r')

def postorder(node):
    global result
    if node * 2 > N: # 자식이 없다
        # 누적 합
        result += lst[node]
    else:
        postorder(node * 2)  # left
        if node * 2 != N:  # 오른쪽 존재
            postorder(node * 2 + 1)  # right


T = int(input())

for tc in range(1, T+1):
    N, M, L = map(int, input().split())

    lst = [0] * (N+1)

    for _ in range(M):
        idx, data = map(int, input().split())
        lst[idx] = data

    result = 0
    postorder(L)
    print('#{} {}'.format(tc, result))

