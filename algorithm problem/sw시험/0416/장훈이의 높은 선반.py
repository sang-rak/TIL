import sys
sys.stdin =


def f(i, n, s, b):
    if i == n: # 모든 사람을 고려한 경우
        if s >= b:
            if minV > s:
                minV = s
    elif minV <=s:
        return

    else:
        f(i+1, n, s + a[i],b)
        f(i+1, n, s,b)


T = int(input())
for tc in range(1,T+1):

