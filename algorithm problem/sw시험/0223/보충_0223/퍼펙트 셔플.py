import sys
sys.stdin = open("퍼펙트 셔플.txt")
T = int(input())

for tc in range(1, T+1):
    N = int(input())
    tmp_list = list(map(str, input().split()))

    # 반 나누기
    cnt = N//2
    if N % 2:  # 총 개수가 홀수일 때
        tmp_list_A = tmp_list[0:cnt]
        tmp_list_B = tmp_list[cnt:N+1]


    else:   # 총 개수가 짝수일 때
        tmp_list_A = tmp_list[0:cnt]
        tmp_list_B = tmp_list[cnt:N + 1]


    result = []
    while tmp_list_A:
        # B 먼저 쌓기
        result.append(tmp_list_B[-1])
        tmp_list_B.pop()
        # A 쌓기
        result.append(tmp_list_A[-1])
        tmp_list_A.pop()

    # 순서 뒤집기
    result.reverse()

    # A는 끝났지만 B는 있다면
    if tmp_list_B:
        result.append(tmp_list_B[-1])
        tmp_list_B.pop()

    print("#{}".format(tc),end=" ")
    for i in range(N):
        print(result[i],end=" ")
    print()


