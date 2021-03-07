import sys
sys.stdin = open('토너먼트 카드게임.txt', 'r')


def win(x, y):
    if arr[x-1] == arr[y-1]:
        return x
    if arr[x-1] == 1 and arr[y-1] == 3:
        return x
    if arr[x-1] == 3 and arr[y-1] == 1:
        return y
    if arr[x-1] > arr[y-1]:
        return x

    return y


def match(start, end):
    if start == end:
        return start
    first = match(start, (start+end)//2)
    second = match((start+end)//2+1, end)
    return win(first, second)


T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = list(map(int, input().split()))
    start = 1
    end = N
    print("#{} {}".format(tc, match(start, end)))