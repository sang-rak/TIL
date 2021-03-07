import sys
sys.stdin = open('Forth.txt', 'r')


def cal():

    stack = []
    for i in range(len(arr)):

        if arr[i] == '+':
            try:
                stack2 = stack.pop()
                stack1 = stack.pop()
                stack3 = int(stack1) + int(stack2)
                stack.append(str(stack3))
            except:
                return 'error'
        elif arr[i] == '-':
            try:
                stack2 = stack.pop()
                stack1 = stack.pop()
                stack3 = int(stack1) - int(stack2)
                stack.append(str(stack3))

            except:
                return 'error'
        elif arr[i] == '/':
            try:
                stack2 = stack.pop()
                stack1 = stack.pop()
                stack3 = int(stack1) // int(stack2)
                stack.append(str(stack3))

            except:
                return 'error'

        elif arr[i] == '*':
            try:
                stack2 = stack.pop()
                stack1 = stack.pop()
                stack3 = int(stack1) * int(stack2)
                stack.append(str(stack3))
            except:
                return 'error'

        elif arr[i] == '.':
            if len(stack) != 1:
                return 'error'
            return stack[0]
        else:
            stack.append(arr[i])

T = int(input())

for tc in range(1, T+1):
    arr = list(map(str, input().split()))
    result = cal()
    print("#{} {}".format(tc, result))