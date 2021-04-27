
import sys
sys.stdin = open('사칙연산.txt', 'r')


def postorder(node):
    if node <= N and num[node] == 0:
        n1 = postorder(fc[node])
        n2 = postorder(sc[node])

        if op[node] == '+':
            num[node] = n1 + n2
        elif op[node] == '-':
            num[node] = n1 - n2
        elif op[node] == '*':
            num[node] = n1 * n2
        elif op[node] == '/':
            num[node] = n1 / n2
    return num[node]
T = 10

for tc in range(1, T+1):

    N = int(input())
    op = [0] * (N+1)
    fc = [0] * (N+1)
    sc = [0] * (N+1)
    num = [0] * (N+1)
    for i in range(N):
        temp = list(input().split())
        if len(temp) == 4:
            op[int(temp[0])] = temp[1]
            fc[int(temp[0])] = int(temp[2])
            sc[int(temp[0])] = int(temp[3])
            num[int(temp[0])] = 0
        else:
            num[int(temp[0])] = int(temp[1])

    postorder(1)

    print("#{} {}".format(tc, int(num[1])))