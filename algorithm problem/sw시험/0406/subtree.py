import sys
sys.stdin = open('subtree.txt', 'r')

T = int(input())

def inorder(node):
    global cnt
    if node == 0:
        return
    cnt += 1
    inorder(leftChild[node])
    inorder(rightChild[node])

for tc in range(1, T+1):
    E, N = map(int, input().split())
    temp = list(map(int, input().split()))
    # parent = [0] * (E+2)
    leftChild = [0] * (E+2)
    rightChild = [0] * (E+2)

    for i in range(E):
        n1, n2 = temp[i*2], temp[i*2+1]
        if leftChild[n1] == 0:
            leftChild[n1] = n2
        else:
            rightChild[n1] = n2
        # parent[n2] = n1


    # print(parent)
    cnt = 0
    inorder(N)
    print('#{} {}'.format(tc, cnt))
