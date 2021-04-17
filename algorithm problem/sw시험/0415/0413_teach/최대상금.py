import sys
sys.stdin = open('최대 상금.txt', 'r')


def swap(prize, i, j):
    # i to a
    numArr = [0] * num0fcard
    for k in range(num0fcard-1, -1, -1):
        numArr[k] = prize % 10
        prize //= 10

    # swap
    numArr[i], numArr[j] = numArr[j], numArr[i]

    # a to i
    prize = 0
    for k in range(num0fcard):
        prize = prize * 10 + numArr[k]

    return prize


def findMax(prize, n, k):  # 숫자판, 교환 횟수, 뎁스
    global ans
    # 메모이제이션 및 가지치기
    for i in range(MAXSIZE):
        if memo[k][i] == prize:  # 가지치기
            return
        elif memo[k][i] == 0:  # 저장
            memo[k][i] = prize
            break



    if n == k:  # 기본파트
        if prize > ans: ans = prize
    else:  # 유도파트
        # num0fcard 중에서 2개 고르는 조합
        for i in range(0, num0fcard-1):
            for j in range(i+1, num0fcard):
                findMax(swap(prize, i, j), n, k+1)

MAXSIZE = 720
T = int(input())

for tc in range(1, T+1):
    prize, num = map(int, input().split())  # 숫자판, 교환횟수
    memo = [[0] * MAXSIZE for _ in range(num+1)]
    num0fcard = 0                           # 숫자판의 자릿수
    ans = 0
    t = prize

    # 숫자판의 자리수 구하기
    while t:
        t //= 10
        num0fcard += 1

    findMax(prize, num, 0)

    print("#{} {}".format(tc, ans))
