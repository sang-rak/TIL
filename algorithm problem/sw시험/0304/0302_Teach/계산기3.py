import sys
sys.stdin = open("계산기3.txt", 'r')

def priority(c):
    if c == '(': return 0
    elif c == '+' or c == '-': return 1
    elif c == '*' or c == '/': return 2



def infix_to_posfix(infix):
    stack = []
    rst_str = []

    # 중위식 스캔
    for i in range(len(infix)):
        # 피연산자 -> 문자열 저장
        if '0' <= infix[i] <= '9':
            rst_str.append(infix[i])

        # 연산자
        else:
            # (: push
            if infix[i] == '(':
                stack.append(infix[i])
            # ): ( 나올때까지 pop -> 문자열
            elif infix[i] == ')':
                while stack[-1] != '(':
                    rst_str.append(stack.pop())
                stack.pop()
            # 사칙연산자
            else:
                if len(stack) != 0:
                    # 중위식 문자(토큰) <= stack[-1]
                    while priority(infix[i]) <= priority(stack[-1]):
                        # else: 토큰보다 우선순위 낮은 연산자 나올때까지 pop -> 문자열
                        rst_str.append(stack.pop())
                        if len(stack) == 0: break  # empty check

                # stack이 비어있거나 우선순위가 더 높으면
                stack.append(infix[i])
    # stack에 값이 남아 있으면
    while len(stack) != 0:
        rst_str.append(stack.pop())
    return "".join(rst_str) # list -> str


def calc(postfix):
    stack = []
    for i in range(len(postfix)):
        # 피연산자: push
        if '0' <= postfix[i] <='9':
            stack.append(int(postfix[i]))
        # 연산자: 2개 pop -> 계산 -> push

        elif postfix[i] == '+':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1 + op2)

        elif postfix[i] == '-':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1 - op2)

        elif postfix[i] == '*':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1 * op2)

        elif postfix[i] == '/':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1 / op2)
    return stack.pop()

T = 10
for tc in range(1, T+1):


    N = int(input())     # 문자열의 크기
    infix = list(input())# 중위식
    postfix = infix_to_posfix(infix)  # 후위식
    print("#{} {}".format(tc, calc(postfix)))


