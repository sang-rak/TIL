import sys
sys.stdin = open('연산.txt', 'r')

from collections import deque


def calc(num, result):
    if result == 0:
        return num+1
    elif result == 1:
        return num-1
    elif result == 2:
        return num*2
    elif result == 3:
        return num - 10



def bfs(N, M):
    global tc
    Q = deque()
    Q.append((N, 0))
    num_lst[N] = tc
    while Q:
        num, cnt = Q.popleft()
        for i in range(4):
            next_num = calc(num, i)
            if next_num == M:
                return cnt + 1
            elif 1 <= next_num <= 1000000 and num_lst[next_num] != tc:
                Q.append((next_num, cnt+1))
                num_lst[next_num] = tc

T = int(input())
num_lst = [0] * (1000000 + 1)
for tc in range(1, T+1):
    N, M = map(int, input().split())
    print("#{} {}".format(tc, bfs(N, M)))
