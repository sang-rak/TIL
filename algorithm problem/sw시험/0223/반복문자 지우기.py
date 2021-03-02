import sys
sys.stdin = open("반복문자 지우기.txt")

T = int(input())
for tc in range(1,T+1):
    tmp = list(str(input()))
    # 처음값 추가
    tmp_list =[tmp[0]]

    for i in range(1, len(tmp)):

        # 공백일 때
        if not tmp_list:
            tmp_list.append(tmp[i])

        # 중복 제거
        elif tmp_list[-1] == tmp[i]:
            tmp_list.pop()

        else:
            tmp_list.append(tmp[i])

    print("#{} {}".format(tc,len(tmp_list)))