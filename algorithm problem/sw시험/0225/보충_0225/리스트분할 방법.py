import sys
sys.stdin = open("스도쿠 검증.txt","r")

T = int(input())

# 2차 배열 에서 사각 영역을 표현하는 방법
# 1. 사각영역의 좌상단 좌표, 우하단 좌표
# 2. 사각영역의 좌상단 좌표, 높이, 너비

arr = [[0] * 10 for _ in range(10)]
r = c = 4  # 좌상단 좌표
h = 3; w = 4

for i in range(r, r+h):
    for j in range(c, c + w):
        arr[i][j] = 1

for lst in arr:
    print(*lst)
