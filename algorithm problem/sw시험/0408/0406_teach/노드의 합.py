import sys
sys.stdin = open('노드의 합.txt', 'r')

def postOrder(node):
    if node <= N:   # 유효한 노드
        # 잎노드의 경우
        if tree[node]:
            return tree[node]
        # 가지노드의 경우
        else:
            # 왼쪽, 오른쪽 받아서 계산하고 리턴
            l = postOrder(2*node)
            r = postOrder(2*node + 1)
            tree[node] = l + r
            return tree[node]
    else:
        return 0


T = int(input())

for tc in range(1, T+1):
    N, M, L = map(int, input().split()) # 노드수, 리프수, 출력노드
    tree = [0] * (N+1)

    for i in range(M):
        idx, value = map(int, input().split())
        tree[idx] = value

    postOrder(1)
    print("#{} {}".format(tc, tree[2]))