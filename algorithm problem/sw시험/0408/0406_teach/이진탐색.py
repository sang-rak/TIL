'''
3
6
8
15
'''
import sys
sys.stdin = open('이진탐색.txt', 'r')

def inorder(node):
    global cnt
    if node <= N:  # 안전 이진트리 정점수와 비교
        inorder(2 * node)
        tree[node] = cnt
        cnt += 1
        inorder(2 * node + 1)


T = int(input())

for tc in range(1, T + 1):
    N = int(input())  # 마지막 정점번호(1 ~ N)
    tree = [0] * (N + 1) # 완전 이진트리 리스트

    cnt = 1  # 포화 이진트리의 번호
    inorder(1)

    print('#{} {} {}'.format(tc, tree[1], tree[N//2]))
