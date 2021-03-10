import sys
sys.stdin = open('회전.txt','r')

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))

    for _ in range(M):
        temp = arr.pop(0)
        arr.append(temp)
    print("#{} {}".format(tc, arr[0]))

'''
import collections

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    deq = collections.deque(arr)
    for _ in range(M):
        temp = deq.popleft()
        deq.append(temp)
    print("#{} {}".format(tc,deq[0]))
'''