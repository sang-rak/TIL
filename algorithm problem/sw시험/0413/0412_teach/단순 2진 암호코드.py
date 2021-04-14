import sys
sys.stdin = open('단순 2진 암호코드.txt', 'r')

# 7차원 리스트의 매핑테이블
code = [[[[[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)]
code[0][0][0][1][1][0][1] = 0  # 0
code[0][0][1][1][0][0][1] = 1  # 1
code[0][0][1][0][0][1][1] = 2  # 2
code[0][1][1][1][1][0][1] = 3  # 3
code[0][1][0][0][0][1][1] = 4  # 4
code[0][1][1][0][0][0][1] = 5  # 5
code[0][1][0][1][1][1][1] = 6  # 6
code[0][1][1][1][0][1][1] = 7  # 7
code[0][1][1][0][1][1][1] = 8  # 8
code[0][0][0][1][0][1][1] = 9  # 9

def findPos(arr):  # 뒤에서 1찾기
    for i in range(r):  # 행
        for j in range(c-1, -1, -1): # 열은 뒤에서 부터
            if arr[i][j] == 1:
                return (i, j)


T = int(input())

for tc in range(1, T+1):
    r, c = map(int, input().split())
    arr = [list(map(int, input())) for _ in range(r)]

    sx, sy = findPos(arr)
    sy -= 55

    # 암호코드 찾기
    p_code = []
    for i in range(8):
        p_code.append(code[arr[sx][sy]][arr[sx][sy+1]][arr[sx][sy+2]][arr[sx][sy+3]][arr[sx][sy+4]][arr[sx][sy+5]][arr[sx][sy+6]])
        sy += 7

    # 암호 코드 검증
    p_value = (p_code[0] + p_code[2] + p_code[4] + p_code[6]) * 3 + p_code[1] + p_code[3] + p_code[5] + p_code[7]
    if p_value % 10 == 0:
        print('#{} {}'.format(tc, sum(p_code)))
    else:
        print('#{} {}'.format(tc, 0))

