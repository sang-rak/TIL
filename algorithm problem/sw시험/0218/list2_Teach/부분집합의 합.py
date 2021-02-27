'''
3
3 6
5 15
5 10
'''
import sys

sys.stdin = open("부분집합의 합_input.txt", "r")

T = int(input())
arr = list(range(1, 13))
n = len(arr)
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for tc in range(1, T + 1):

    N, K = map(int, input().split()) # 부분집합 원소의 개수, 합

    ans = 0

    for i in range(1 << n):  # 1<<n:부분 집합의 개수
        sum = 0
        cnt = 0
        for j in range(n):  # 원소의 수만큼 비트를 비교함
            if i & (1 << j):  # i의 j번째 비트가 1이면 j번째 원소 출력
                sum += arr[j]
                cnt += 1

        if sum == K and cnt == N:
            ans += 1

    print('#{} {}'.format(tc, ans))