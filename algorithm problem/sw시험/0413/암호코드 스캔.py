import sys
sys.stdin = open('암호코드 스캔.txt', 'r')

Conversion = {'0':'0000', '1':'0001', '2':'0010', '3':'0011',
         '4':'0100', '5':'0101', '6':'0110', '7':'0111',
         '8':'1000', '9':'1001', 'A':'1010', 'B':'1011',
         'C':'1100', 'D':'1101', 'E':'1110', 'F':'1111'}

decryption = {'211':0, '221':1, '122':2, '411':3, '132':4, '231':5, '114':6, '312':7, '213':8, '112':9}

def reduce(c, b, a):
    min_num = min(c,b,a)
    c //= min_num
    b //= min_num
    a //= min_num
    return str(c)+str(b)+str(a)

TC = int(input())
for tc in range(1, TC+1):
    N, M = map(int, input().split())
    Scannner = [input() for _ in range(N)]

    Binary_lst = [''] * N
    for i in range(N):
        for j in range(M):
            Binary_lst[i] += Conversion[Scannner[i][j]]
    # print(Binary_lst)

    result = []
    visited = []
    ans = 0
    for y in range(N):
        a = b = c = 0
        for x in range(M*4-1, -1, -1):
            if b == 0 and c == 0 and Binary_lst[y][x] == '1':
                a += 1
            elif a > 0 and c == 0 and Binary_lst[y][x] == '0':
                b += 1
            elif a > 0 and b > 0 and Binary_lst[y][x] == '1':
                c += 1

            if a > 0 and b > 0 and c > 0 and Binary_lst[y][x] == '0':
                result.append(decryption[reduce(c, b, a)])
                a = b = c = 0

            if len(result) == 8:
                result = result[::-1]
                value = (result[0] + result[2] + result[4] + result[6]) * 3 + \
                        (result[1] + result[3] + result[5]) + result[7]

                if value % 10 == 0 and result not in visited:
                    ans += sum(result)

                visited.append(result)
                result = []

    print('#%d %d'%(tc, ans))

    # 필요한 곳 뺴내기 뒤에서부터 1이고 위쪽 칸의 수가 0인 j지점 뽑아내기
    # 1. 16진수 -> 2진수 저장(N, M * 4)
    # for i: N
    #     j: M * 4 - 1
    #       while j >= 55:
    #             if  arr[i][j] == 1 and arr[i-1][j] == 0:
    #                 while arr[i][j] == 1:
    #                     z += 1
    #                     j -= 1
    #                 while arr[i][j] == 0:
    #                     y += 1
    #                     j -= 1
    #                 while arr[i][j] == 1:
    #                     x += 1
    #                     j -= 1
    #                 while
    #                     j -= 1
    #
    #         j += 1

    # while arr[i][j] ==1:
    # z += 1
    # j -= 1

    # while arr[i][j] ==0:
    # z += 1
    # j -= 1

    # while arr[i][j] ==1:
    # z += 1
    # j -= 1

    # while arr[i][j] ==0:
    # z += 1
    # j -= 1

    # min ( x, y, z)
    # 비율 찾기

    # for i : 0~N
    #       j = M+4 -1 열순환
    #         while j > 55:
    #             if arr[i][j] == 1 and arr[i-1][j] ==0:
    #
    #             for k : 0~8 암축
    #                 비율 x, y, z)
    #