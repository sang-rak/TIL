# S: 스페이드, D: 다이아몬드, H: 하트, C:클로버
# S01D02H03H04
# 1. 입력을 받아서 가지고 있는 카드정보들을 잘라서 가져온다.
#   - 3자리 끊어서 읽는다. for i in range(0, N, 3):
# 2. 카드 정보를 식별 --> 무늬, 숫자
#   - 무늬 - 문자 arr[i] / 숫자 -> 문자열을 정수로 변환
#   int(arr[i + 1]) * 10 + int(arr[i + 2])
#   int(arr[i + 1:i + 3])
# 3. 하나씩 카드정보를 저장 --> 중복체크, 출력할 내용(무늬별 카드수)
#  무늬를 숫자로 변환(0, 1, 2, 3), 숫자(1 ~ 13)
# 2차 배열 --> cnt = [[0] * 14 for _ in range(4)]

arr = 'S01D02H03H04'

def getPattern(p):
    if p == 'S': return 0
    if p == 'D': return 1
    if p == 'H': return 2
    if p == 'C': return 3

cnt = [[0] * 14 for _ in range(4)]
flag = True
for i in range(0, len(arr), 3):
    p = getPattern(arr[i])
    num = int(arr[i + 1]) * 10 + int(arr[i + 2])

    if cnt[p][num] == 1:
        flag = False
        break
    else:
        cnt[p][num] = 1

# 카드 개수 확인
ans = [0] * 4
for i in range(4):
    for j in range(1, 13 + 1):
        if cnt[i][j] == 0:
            ans[i] += 1