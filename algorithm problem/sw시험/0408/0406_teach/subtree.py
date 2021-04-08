import sys
sys.stdin = open('subtree.txt', 'r')

def preorder(node):
    global cnt
    if node != 0:
        cnt += 1                 # 하고 싶은일
        preorder(tree[node][0])  # 왼쪽 자식
        preorder(tree[node][1])  # 오른쪽 자식



T = int(input())

for tc in range(1, T+1):
    E, N = map(int, input().split())  # 간선수, 시작노드
    tree = [[0] * 3 for _ in range(E+2)]  # 왼쪽, 오른쪽, 부모
    temp = list(map(int, input().split()))
    cnt = 0   # 서브트리의 노드의 수
    for i in range(E):
        p = temp[i*2]
        c = temp[i*2+1]
        if tree[p][0] == 0:
            tree[p][0] = c
        else:
            tree[p][1] = c
        tree[c][2] = p

    preorder(N)
    print("#{} {}".format(tc, cnt))