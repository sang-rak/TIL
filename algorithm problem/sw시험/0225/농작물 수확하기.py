import sys
sys.stdin = open("농작물 수확하기.txt", "r")

T = int(input())

for tc in range(1, T+1):

    N = int(input())

    # arr 리스트형식 저장
    arr = []
    for i in range(N):
        arr_list = [0] * N
        tmp = list(map(str, input().split()))
        for j in range(N):
            arr_list[j] = int(tmp[0][j])
        arr.append(arr_list)


    # 계산
    cnt = 0
    for i in range(N):
        for j in range(N):
            # 위쪽 절반줄 까지
            if i <= (N//2):
                # 가운데줄먹기
                if N//2-i <= j <= N//2+i:
                    cnt += arr[i][j]
                    # print("#",arr[i][j])

            else: # 아래쪽 center 줄 밑으로
                if i-N//2 <= j <= N+N//2-i-1:
                    cnt += arr[i][j]
                    # print("&", arr[i][j])

                pass
    print("#{} {}".format(tc, cnt))