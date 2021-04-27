import sys
sys.stdin = open('중위순회.txt', 'r')

def inorder(node):
    if node <= N:
        inorder(node * 2)          # left
        print(tree[node], end='')  # root
        inorder(node * 2 + 1)      # right

T = 10

for tc in range(1, T+1):

    N = int(input())
    tree = [[0] for _ in range(N + 1)]

    for i in range(N):
        temp = input().split()
        tree[int(temp[0])] = temp[1]
    print('#{} '.format(tc), end='')
    inorder(1)
    print()