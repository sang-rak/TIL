import sys
sys.stdin = open('베이비진 게임.txt', 'r')

T = int(input())

for tc in range(1, T+1):

    arr = list(map(int, input().split()))


    # p1, p2 카드나누기
    p1 = []
    p2 = []
    for i in range(len(arr)//2):
        p1.append(arr[i * 2])
    for i in range(len(arr)//2):
        p2.append(arr[i * 2 + 1])


    p1_stack = []
    p2_stack = []
    win_p1 = 0
    win_p2 = 0
    cnt = 0
    for i in range(6):
        cnt += 1
        # 3개를 유지하는 스택
        p1_stack.append(p1[i])
        p2_stack.append(p2[i])
        if len(p1_stack) > 3:
            del p1_stack[0]
        if len(p2_stack) > 3:
            del p2_stack[0]
        if cnt >= 3:
            sort_p1 = sorted(p1_stack)
            sort_p2 = sorted(p2_stack)
            print(sort_p2)
            # 연속인숫자가 3개 stack2 = (stack1+stack3)//2
            if sort_p1[0] == (sort_p1[1]+sort_p1[2])//2:
                win_p1 = 1
            if sort_p2[0] == (sort_p2[1]+sort_p2[2])//2:
                win_p2 = 1
            # 같은숫자가 3개 stack2 = stack1 and stack1 = stack3)
            if sort_p1[0] == sort_p1[1] and sort_p1[1] == sort_p1[2]:
                win_p1 = 1
            if sort_p2[0] == sort_p2[1] and sort_p2[1] == sort_p2[2]:
                win_p2 = 1


            if win_p1 == 1 and win_p2 == 1:
                print('#{} {}'.format(tc, 0))
            elif win_p1 == 1:
                print('#{} {}'.format(tc, 1))
            elif win_p2 == 1:
                print('#{} {}'.format(tc, 2))
