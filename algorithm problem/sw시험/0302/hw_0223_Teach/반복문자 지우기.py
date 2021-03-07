import sys
sys.stdin = open("반복문자 지우기.txt")

T = int(input())

for tc in range(1, T+1):
    stack = []
    str = input()

    for i in range(len(str)):
        #  스택이 비어있으면 push
        if len(stack) == 0:
            stack.append(str[i])
        else:
            if stack[-1] == str[i]:
                stack.pop()
            else:
                stack.append(str[i])
    print("{} {}".format(tc, len(stack)))

