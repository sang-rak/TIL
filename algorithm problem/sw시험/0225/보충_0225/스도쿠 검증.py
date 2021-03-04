import sys
sys.stdin = open("스도쿠 검증.txt","r")

T = int(input())

# 2차 배열 에서 사각 영역을 표현하는 방법
# 1. 사각영역의 좌상단 좌표, 우하단 좌표
# 2. 사각영역의 좌상단 좌표, 높이, 너비



def row_check():

    # 행 단위로 체크
    for r in range(9):
        check = [0] * 10  # 1~9 인덱스를 사용
        for c in range(9):
            if check[arr[r][c]]:
                return 0
            check[r][c] = 1
    return 1
    # 열 단위로 체크
    for r in range(9):
        check = [0] * 10  # 1~9 인덱스를 사용
        for c in range(9):
            if check[arr[c][r]]:
                return 0
            check[arr[c][r]] = 1
    return 1

    # 3x3 사각영역 체크
    def rect_check():
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):

                # r,c -> 좌상단 좌표
                check = [0] * 10
                for i in range(r, r + 3):
                    for j in range(c, c + 3):
                        if check[arr[i][j]]:
                            return 0
                        check[arr[r][c]]


for tc in range(1, int(input()) + 1):
    arr = [list(map(int, input().split())) for _ in range(9)]

    if row_check() and col_check() and rect_check():

