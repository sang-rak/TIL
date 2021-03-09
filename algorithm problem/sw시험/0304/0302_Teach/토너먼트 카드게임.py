import sys
sys.stdin = open('토너먼트 카드게임.txt', 'r')

def f(l, r): # 왼쪽, 오른쪽 인덱스
    # 기본파트 (깊이)
    if l == r:
        return l
    # 유도파트
    else:
        # 아래서 받은 값
        r1 = f(l, (l+r)//2)
        r2 = f((l+r)//2+1, r)
        # 계산 해서 리턴
        if card[r1] == card[r2]:
            return r1
        if card[r1] == 1 and card[r2] == 3:
            return r1
        if card[r1] == 3 and card[r2] == 1:
            return r2
        if card[r1] > card[r2]:
            return r1

        return r2



T = int(input())

for tc in range(1, T+1):
    N = int(input())
    card = [0] + list(map(int, input().split()))

    print("#{} {}".format(tc, f(1,N)))