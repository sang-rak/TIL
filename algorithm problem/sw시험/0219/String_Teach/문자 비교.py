import sys
sys.stdin = open("문자열 비교_input.txt", 'r')

T = int(input())

# def brout_force1(p, t):
#
#     #for 문
#     for i in range(len(t)-len(p)+1):
#         # 패턴의 길이만큼 반복
#         for j in range(len(p)):
#             if p[j] != t[i+j]:
#                 break
#             else:
#                 return 1
#     return 0


def brout_force2(p, t):
    i = 0
    j = 0

    while j < len(p) and i < len(t):
        if p[j] != t[i]:
            i = i-j
            j = -1

        i += 1
        j += 1

    # 패턴을 찾았다.
    if j == len(p): return 1
    else: return 0


for tc in range(1, T+1):
    str1 = str(input())
    str2 = str(input())
    result = burt(str1, str2)
    print('#{} {}'.format(tc,result))