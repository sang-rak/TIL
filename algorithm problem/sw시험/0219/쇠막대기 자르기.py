import sys
sys.stdin = open("쇠막대기 자르기.txt")

T = int(input())

for tc in range(1, T+1):
    still = input()

    cnt = 0
    ans = 0
    for i in range(len(still)):
        if still[i] == '(':
            cnt += 1
        else:
            cnt -= 1
            if still[i-1] == '(':
                ans += cnt
            else:
                ans += 1
    print("#{} {}".format(tc,ans) )