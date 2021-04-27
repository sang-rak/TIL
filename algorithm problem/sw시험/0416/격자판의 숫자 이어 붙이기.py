import sys
sys.stdin = open('격자판의 숫자 이어 붙이기.txt', 'r')

def find(i,j,n,s):
    if n == 7:
        t.add(s)
    else:
        for k in range(4):
            ni = i + di[k]
            nj = j + dj[k]
            if (ni >= 0 and ni < 4 and nj >= 0 and nj < 4):
                find(ni, nj, n+1, s+str(a[ni][nj]))

T = int(input())

for tc in range(1, T+1):

    a = [list(map(int, input().split())) for _ in range(4)]
    di = [0, 0, 1, -1]
    dj = [1, -1, 0 ,0]
    t = set()
    for i in range(4):
        for j in range(4):
            find(i,j,0,'')
    print('#{} {}'.format(tc, len(t)))