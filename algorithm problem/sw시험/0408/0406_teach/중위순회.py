import sys
sys.stdin = open('중위순회.txt', 'r')

def inorder(node):
    if node != 0:
        inorder(firstChild[node])
        print(alpha[node], end='')
        inorder(secondChild[node])



T = 10

for tc in range(1, T+1):

    N = int(input())            # 정점의 갯수
    firstChild = [0] * (N+1)    # 왼쪽자식
    secondChild = [0] * (N+1)   # 오른쪽 자식
    alpha = [''] * (N+1)        # 알파벳
    for i in range(N):
        temp = list(input().split())
        addr = int(temp[0])          # 정점의 번호
        alpha[addr] = temp[1]
        if addr * 2 <= N:     # 왼쪽 자식
            firstChild[addr] = int(temp[2])
            if addr * 2 + 1 <= N :  # 오른쪽 자식
                secondChild[addr] = int(temp[3])
    print('#{}'.format(tc), end='')
    inorder(1)
    print()
