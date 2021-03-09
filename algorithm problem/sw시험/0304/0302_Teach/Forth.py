import sys
sys.stdin = open('Forth.txt', 'r')


def calc(exp):
    stack = []
    # 문자열 스캔
    for i in range(len(exp)):
        # 연산자 일때 ( 피연산자 2개 이상)
        if exp[i] == '+' or exp[i] == '-' or exp[i] == '/' or exp[i] == '*':
            if len(stack) >= 2:
                op2 = int(stack.pop())  # 두번째 피연산자
                op1 = int(stack.pop())  # 첫번째 피연산자
                if exp[i] == '+':
                    stack.append(op1 + op2)
                elif exp[i] == '-':
                    stack.append(op1 - op2)
                elif exp[i] == '/':
                    stack.append(op1 // op2)
                elif exp[i] == '*':
                    stack.append(op1 * op2)

            else:
                return "error"
        # 숫자 (!연산자, !빈칸, !.) -> push
        elif exp[i] != ' ' and exp[i] != '.':
            stack.append(exp[i])
        # . 일때
        elif exp[i] == '.':
            if len(stack) == 1:
                return stack.pop()
            else:
                return "error"
T = int(input())

for tc in range(1, T+1):
    exp = input().split()


    print("#{} {}".format(tc, calc(exp)))