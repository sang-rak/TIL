import sys
sys.stdin = open("퍼펙트 셔플.txt")
T = int(input())

for tc in range(1, T+1):
    N = int(input())
    cards = list(input().split())
    ans = []
    l = 0
    r = (N+1) // 2
    for _ in range(N//2):
        ans.append(cards[l])
        ans.append(cards[r])
        l, r = l+1, r+1

    if N % 2:
        ans.append(cards[N//2])

    print('#{} {}'.format(tc, ''.join(map(str, ans))))
