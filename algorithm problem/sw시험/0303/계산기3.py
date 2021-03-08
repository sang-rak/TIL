import sys
sys.stdin = open("계산기3.txt", 'r')

T = 10
for tc in range(1, T+1):
    N = int(input())
    arr = list(input())

    s = []
    num_s = []

    for i in arr:
        if i == '(':
            s.append(i)
        elif i == ')':
            while s[-1] != '(':
                a = s.pop()
                num_s.append(a)
            s.pop()
        elif i == '+':
            while s and s[-1] != '(':
                a = s.pop()
                num_s.append(a)
            s.append(i)
        elif i == '*':
            while s and s[-1] != '(' and s[-1] != '+':
                a = s.pop()
                num_s.append(a)
            s.append(i)
        else:
            num_s.append(int(i))

    while s:
        a = s.pop()
        num_s.append(a)

    for i in num_s:
        if i == '+' or i == '*':
            num1 = s.pop()
            num2 = s.pop()

            if i == '+':
                res = num2 + num1
            else:
                res = num2 * num1

            s.append(res)

        else:
            s.append(i)

    print('#{} {}'.format(tc, *s))

