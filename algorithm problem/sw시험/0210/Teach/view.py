import sys
sys.stdin = open("view_input.txt")
T = 10 # 대문자 상수
for tc in range(1, T+1):
    N = int(input())
    arr = list(map(int, input().split()))
    ans = 0

    # 2 ~ N-2 각각 검사해서
    for i in range(2, N-2):
        min_value = 987654321
        # 기준 건물과 왼쪽 오른쪽 2개 차의 최소값
        for j in range(5):
            if j != 2:
                if arr[i] - arr[i-2+j] < min_value:
                    min_value = arr[i] - arr[i-2+j]
        # 최소값이 양수이면 조망권이 확보
        if min_value > 0 :
            ans += min_value
    print("#{} {}".format(tc, ans))