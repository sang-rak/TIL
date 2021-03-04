import sys
sys.stdin = open("백만장자 프로젝트.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N = int(input())

    arr = list(map(int, input().split()))

    max_idx = -1
    benefit = 0
    # arr 남아있을 때 까지 실행
    while arr:

        # max 값찾기
        max_idx = arr.index(max(arr))

        # max 값 전의 날 다 구입 (비용)
        long_cost = 0
        stock = 0
        for i in range(max_idx):
            long_cost += arr[i]
            stock += 1

        # max 날 팔기 (수익)
        benefit += arr[max_idx] * stock - long_cost

        #  max 날까지 제거
        arr = arr[max_idx + 1::]

    print("#{} {}".format(tc, benefit))




#1 4053
#2 6385
#3 26725
#4 211514
#5 4848198
#6 49761546
#7 500155606
#8 4995241394
#9 4999367498
#10 4995633799
