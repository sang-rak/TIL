import sys
sys.stdin = open("영준이의 카드 카운팅.txt")

T = int(input())

for tc in range(1, T+1):

    def card():
        card_list = []
        card_H = []
        card_S = []
        card_D = []
        card_C = []
        card = str(input())
        for i in range(0,len(card),3):
            card_list += [card[i:i+3]]
        count = 0
        for j in range(len(card_list)):
            if card_list[j][0] == 'H':
                if card_list[j][1:3] not in card_H:
                    card_H += [card_list[j][1:3]]
                else:
                    print("#{} ERROR".format(tc))
                    count = 1

            elif card_list[j][0] == 'S':
                if card_list[j][1:3] not in card_S:
                    card_S += [card_list[j][1:3]]
                else:
                    print("#{} ERROR".format(tc))
                    count = 1

            elif card_list[j][0] == 'D':
                if card_list[j][1:3] not in card_D:
                    card_D += [card_list[j][1:3]]
                else:
                    print("#{} ERROR".format(tc))
                    count = 1

            elif card_list[j][0] == 'C':
                if card_list[j][1:3] not in card_C:
                    card_C += [card_list[j][1:3]]
                else:
                    print("#{} ERROR".format(tc))
                    count = 1

        s = 13 - len(card_S)
        h = 13 - len(card_H)
        c = 13 - len(card_C)
        d = 13 - len(card_D)
        if count != 1:
            print("#{} {} {} {} {}".format(tc, s, d, h, c))

    card()

