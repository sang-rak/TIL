N = 3
card = [4, 5, 6]

# 반복문을 이용해서 작성

#run?
#triplet?

run = False
tri = False
for i in range(N):
    for j in range(N):
        if j != i:
            for k in range(N):
                if k != i and k != j:
                    print(card[i], card[j], card[k])

                    # run 검사
                    if card[i] + 1 == card[j] and card[i] + 2 == card[k]:
                        run = True
                        
                    # # tri 검사 늦음
                    # if card[i] == card[j] and card[i] == card[]:
                    #     tri = True
                    #     
                    # if card[u