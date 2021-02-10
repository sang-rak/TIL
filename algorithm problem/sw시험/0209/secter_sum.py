import sys
sys.stdin = open("secter_input.txt", "r")

T = int(input())
for tc in range(1,T+1):
    N, n = list(map(int, input().split()))
    arr = list(map(int, input().split()))
    # 최대


    # 섹터 합의 개수인 n을 뺀다

    for i in range(0, len(arr)-n+1):
        max_arr = 0
        min_arr = 0
        if i == 0:
            for k in range(n):
                max_arr += arr[i + k]
                min_arr += arr[i + k]
            max_cp = max_arr
            min_cp = min_arr

        else:
            for j in range(n):
                max_arr += arr[i+j]
                min_arr += arr[i+j]

            if max_cp < max_arr:
                max_cp = max_arr

            if min_cp > min_arr:
                min_cp = min_arr
    print(f"#{tc} {max_cp-min_cp}".format(tc, max_cp, min_cp))

