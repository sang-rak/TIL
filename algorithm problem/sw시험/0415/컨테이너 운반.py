import sys
sys.stdin = open('컨테이너 운반.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    # 컨테이너 수 N과 트럭 수 M
    N, M = map(int, input().split())

    block = sorted(list(map(int, input().split())))
    truck = sorted(list(map(int, input().split())))


    result = 0
    for i in range(M):
        check = -1

        for j in range(len(block)):
            if truck[i] >= block[j]:
                check = j

        if check != -1:
            # 더하기
            result += block[check]
            # block 지우기

            del(block[check])

    print('#{} {}'.format(tc, result))
