import sys
sys.stdin = open("문자열 비교_input.txt", 'r')

T = int(input())

def burt(str1, str2):

    N = len(str1)
    M = len(str2)

    for i in range(M-N+1):
        cnt = 0
        for j in range(N):
            if str2[i+j] == str1[j]:
                cnt += 1
        if cnt == N:
            return 1
    return 0

for tc in range(1, T+1):
    str1 = str(input())
    str2 = str(input())
    result = burt(str1, str2)
    print('#{} {}'.format(tc,result))