import sys

sys.stdin = open("색칠하기_input.txt", "r")

def printarr(a):
    for i in range(10):
        for j in range(10):
            print(a[i][j], end=" ")
        print()
    print()

T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for tc in range(1, T + 1):
    arr = [[0] * 10 for i in range(10)]

    cnt = 0
    N = int(input()) # 색칠할 개수
    for _ in range(N):
        r1, c1, r2, c2, color = map(int, input().split())
        # 색칠하기
        for i in range(r1, r2+1):
            for j in range(c1, c2+1):

                arr[i][j] += color
    # 디버깅용
    # printarr(arr)

    # 겹쳐진 칸수 카운팅
    for i in range(10):
        for j in range(10):
            if arr[i][j] == 3: cnt += 1
    print("#{} {}".format(tc, cnt))