'''
3
400 300 350
1000 299 578
1000 222 888
'''
import sys
sys.stdin = open("이진탐색_input.txt", "r")

T = int(input())
def binary_search(a, key, page):
    start = 1
    end = page
    cnt = 0
    while start <= end:
        mid = (start + end)//2
        cnt += 1
        if key == mid: # 성공
            return cnt
        elif key < mid:
            end = mid
        else:
            start = mid
    return -1


for tc in range(1, T + 1):

    P, A, B = map(int, input().split()) # 전체 쪽 수: P 각각찾을 쪽수 a, b
    arr = list(range(P+1))
    # print(arr)
    a = binary_search(arr, A, P)
    b = binary_search(arr, B, P)

    if a > b: ans ='B'
    elif a < b: ans = 'A'
    else: ans = '0'

    print("#{} {}".format(tc, ans))