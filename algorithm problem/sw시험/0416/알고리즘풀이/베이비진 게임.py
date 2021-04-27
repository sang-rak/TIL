import sys
sys.stdin = open('베이비진 게임.txt', 'r')

def game():
    count1 = [0 for i in range(10)]
    count2 = [0 for i in range(10)]
    for i in range(12):
        n = card[i]
        if i%2 == 0:               # player1
            count1[n] += 1
            if count1[n] == 3:     # triplet 검사
                return 1
            if run(count1):        # run 검사
                return 1
        else:
            count2[n] += 1
            if count2[n] == 3:     # triplet 검사
                return 2
            if run(count2):        # run 검사
                return 2
    return 0 # 승자가 없는 경우

def run(count):
    for i in range(8):
        if count[i] >= 1 and count[i+1] >= 1 and count[i+2] >= 1:
            return 1

T = int(input())

for tc in range(1, T+1):

    card = list(map(int, input().split()))  # 12장의 카드 정보
    print("#{} {}".format(tc, game()))