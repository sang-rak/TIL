'''
( )( )((( )))
((( )((((( )( )((( )( ))((( ))))))
'''

def check(arr):
    # 문자열을 스캔
    stack = []
    for i in range(len(arr)):

        # ( : push
        if arr[i] == "(":
            stack.append(arr[i])

        # ) : pop
        elif arr[i] == ")":
            # is Empty -> F
            if len(stack) == 0:
                return False
            else:
                stack.pop()

    # stack not empty -> F
    if len(stack) != 0:
        return False # 비어 있으면 False
    else:
        return True


arr = input()
print(check(arr))