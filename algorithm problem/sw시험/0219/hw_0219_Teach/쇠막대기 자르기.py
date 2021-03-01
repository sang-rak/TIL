import sys
sys.stdin = open("쇠막대기 자르기.txt", "r")

T = int(input())

'''
for tc in range(1, T+1):
    iron_bar = input()

    cnt = 0 # 막대 수
    ans = 0 # 정답

    for i in range(len(iron_bar)):
        # 열린 괄호라면 막대 추가
        if iron_bar[i] == '(':
            cnt += 1
        else:
            # 닫힌 괄호라면 막대감소
            # 레이저라면 당연히 잘못 세었으니까 빼는게 맞다.
            # 아니라면 어차피 철봉 끝이니 빼는게 맞다.
            cnt -= 1

            # 레이저
            if iron_bar[i-1] == '(':
               # 레이저로 인해서 잘린 막대들이 생겼으므로
               ans += cnt
            else:
                # 막대 끝이라는 뜻
                ans += 1

    print("#{} {}".format(tc, ans))
    '''
# stack으로 풀기
for tc in range(1, T+1):
    iron_bar = input()

    #실제로 철봉이 담길 리스트
    s = []
    ans = 0

    for i in range(len(iron_bar)):
        # 열릴 괄호라면 s 리스트에 넣어놓기
        if iron_bar[i] == '(':
            s.append('(')
        else:
            # 무조건 꺼내기
            s.pop()

            if iron_bar[i-1] == '(':
                ans += len(s)
            else:
                ans += 1
    print("#{} {}".format(tc, ans))
