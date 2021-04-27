import sys
sys.stdin = open('사칙연산.txt', 'r')

def calc(op, left, right):
    if op == '+':
        result = left + right
    elif op == '-':
        result = left - right
    elif op == '*':
        result = left * right
    elif op == '/':
        result = left / right

    return result


# 후위순회
def postOrder(node):
    # 리프노드
    if firstChild[node] == 0 or secondChild[node] == 0:
        return num[node]

    # 잎노드
    else:
        l = postOrder(firstChild[node])
        r = postOrder(secondChild[node])
        num[node] = calc(oper[node], l, r)
        return num[node]
T = 10

for tc in range(1, T+1):

    N = int(input())  # 정점의 수
    oper = [0] * (N+1)
    firstChild = [0] * (N+1)
    secondChild = [0] * (N+1)
    num = [0] * (N+1)

    for i in range(N):
        temp = list(input().split())
        idx = int(temp[0])  # 정점번호
        if temp[1] == '+' or temp[1] == '-' or temp[1] == '*' or temp[1] == '/':  # 연산자이면
            oper[idx] = temp[1]
            firstChild[idx] = int(temp[2])
            secondChild[idx] = int(temp[3])
        else:
            num[idx] = int(temp[1])

    ans = postOrder(1)
    print("#{} {}".format(tc, int(ans)))