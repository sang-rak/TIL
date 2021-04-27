'''
3
6
8
15
'''
def inorder(node):
    global cnt
    if node <= N:
        inorder(node * 2)
        tree[node] = cnt
        cnt += 1
        inorder(node * 2 + 1)

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    tree = [0] * (N+1)

    cnt = 1
    inorder(1)

    print('#{} {} {}'.format(tc, tree[1], tree[N//2]))
