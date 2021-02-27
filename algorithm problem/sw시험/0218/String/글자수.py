import sys
sys.stdin = open("글자수_input.txt","r")

T = int(input())
for tc in range(1, T+1):
    # 입력
    str1 = str(input())
    str2 = str(input())


    # 첫번째 리스트를 같으면 1 다르면 딕셔너리 만들기
    str_list = {}
    for i in range(len(str1)):
        str_list[str1[i]] = 0

    # 두번째 리스트 딕셔너리를 호출 후 나오는 값에 +1 하고 돌려주기
    for j in range(len(str2)):
        try:
            if str_list[str2[j]] >= 0:
                str_list[str2[j]] += 1
        except: # 값이 없어서 나는 오류처리
            continue
    # sort해서 가장 높은값을 뽑아내기
    str_list = sorted(str_list.values())
    print("#{} {}".format(tc, str_list[-1]))

