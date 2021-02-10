'''
9
7 4 2 0 0 6 0 7 0
'''


n = int(input())   # 상자 높이
arr = list(map(int, input().split()))
result = 0 #낙차의 최대값
# maxHeight = 0 # i의 최대낙차값

for i in range(len(arr)):  # 상자들의 높이 검사
    maxHeight = len(arr)-i-1  # i의 최대낙차값: 9
    for j in range(i+1, len(arr)):  # i보다 큰 상자들과 비교
        if arr[i] <= arr[j]:  # 최대낙차값에서 1씩 감소
            maxHeight -= 1

    # 최대값유지
    if result < maxHeight:
        result = maxHeight

print(result)