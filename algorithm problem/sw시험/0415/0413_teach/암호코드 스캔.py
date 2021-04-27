import sys

sys.stdin = open("암호코드 스캔.txt")
asc = [[0, 0, 0, 0],  # 0
       [0, 0, 0, 1],  # 1
       [0, 0, 1, 0],  # 2
       [0, 0, 1, 1],  # 3
       [0, 1, 0, 0],  # 4
       [0, 1, 0, 1],  # 5
       [0, 1, 1, 0],  # 6
       [0, 1, 1, 1],  # 7
       [1, 0, 0, 0],  # 8
       [1, 0, 0, 1],  # 9
       [1, 0, 1, 0],  # A
       [1, 0, 1, 1],  # B
       [1, 1, 0, 0],  # C
       [1, 1, 0, 1],  # D
       [1, 1, 1, 0],  # E
       [1, 1, 1, 1]]  # F

scode = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(5)]
scode[2][1][1] = 0
scode[2][2][1] = 1
scode[1][2][2] = 2
scode[4][1][1] = 3
scode[1][3][2] = 4
scode[2][3][1] = 5
scode[1][1][4] = 6
scode[3][1][2] = 7
scode[2][1][3] = 8
scode[1][1][2] = 9


def solve():
    ret = 0
    # 종료점 찾기
    for i in range(N):
        j = M * 4 - 1  # 열의 마지막 인덱스
        while j >= 55:
            if arr[i][j] == 1 and arr[i - 1][j] == 0:
                # 비율 찾기
                code = [0] * 8  # 암호코드
                for k in range(7, -1, -1):
                    x = y = z = 0  # 1 : 0 : 1  개수저장
                    while arr[i][j] == 1:  # 1의 갯수
                        z += 1
                        j -= 1
                    while arr[i][j] == 0:  # 0의 갯수
                        y += 1
                        j -= 1
                    while arr[i][j] == 1:  # 1의 갯수
                        x += 1
                        j -= 1
                    while arr[i][j] == 0:  # 0의 갯수
                        j -= 1

                    d = min(x, y, z)
                    x //= d
                    y //= d
                    z //= d

                    code[k] = scode[x][y][z]

                # 암호값 검증
                t = (code[0] + code[2] + code[4] + code[6])*3 + code[1] + code[3]+code[5]+ code[7]
                if t % 10 == 0:
                    ret += code[0] + code[2] + code[4] + code[6] + code[1] + code[3]+code[5]+ code[7]

                j += 1 # 같은행에 다른 암호코드 겹쳐 있으면 다른 암호코드의 마지막에 1에 도착
            j -= 1 # while에서 감소
    return ret

T = int(input())
for tc in range(1, T + 1):
    N, M = map(int, input().split())  # 행렬
    arr = [[0] * (M * 4) for _ in range(N)]

    # 16진수를 2진수로 변환하여 입력받기
    for i in range(N):
        temp = input()
        for j in range(M):
            ch = temp[j]  # 16진수 한글자 저장
            if ch <= '9':
                num = ord(ch) - ord('0')  # num은 10진수
            else:
                num = ord(ch) - ord('A') + 10
            for k in range(4):
                arr[i][j * 4 + k] = asc[num][k]
    ans = solve()
    print("#{} {}".format(tc, ans))
