'''
3
400 300 350
1000 299 578
1000 222 888
'''
import sys

sys.stdin = open("이진탐색_input.txt", "r")

T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):

    P, Pa, Pb = map(int, input().split()) # 전체 쪽 수: P 각각찾을 쪽수 a, b
    def cent(P,Pa):
        start = 1
        end = P
        count = 1

        center = (start + end) // 2

        while center != Pa:
            count += 1
            if center < Pa:
                start = center
            else:
                end = center
            center = (start + end) // 2

        return count
    if cent(P, Pa) < cent(P, Pb):
        print('#{} A'.format(test_case))
    elif cent(P, Pa) == cent(P, Pb):
        print('#{} 0'.format(test_case))
    else:
        print('#{} B'.format(test_case))

    # 1 A
    # 2 0
    # 3 A