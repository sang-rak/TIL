import sys
sys.stdin = open("글자수_input.txt","r")

# for

T = int(input())
for tc in range(1, T+1):
    # 입력
    str1 = str(input())
    str2 = str(input())

    # str1 각 문자에 카운트를 세기 위함
    cnt = [0] * len(str1)

    # str1
    for i in range(len(str1)):
        # str1[i] 가 str2에 몇개 들어있는지 체크
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                cnt[i] += 1
    ans = 0

    #가장 큰값 찾기
    for i in range(len(cnt)):
        if ans < cnt[i]:
            ans = cnt[i]
    print("#{} {}".format(tc, ans))

# 딕셔너리
'''
T = int(input())

for tc in range(1, T+1):
    str1 = input()
    str2 = input()

    # 키 = 문자, value = 카운트한 수
    my_dict = {}

    # str1의 문자를 가진 딕셔너리 생성
    for key in set(str1):
        my_dict[key] = 0

    for key in str2:
        if key in my_dict:
            my_dict[key] += 1

    ans = 0

    for i in my_dict.values():
        if ans < i:
            ans = i
    print("#{} {}".format(tc, ans))
'''
'''
# 만약 주어진 문자가 모두 대문자만 들어온다면
for tc in range(1, int(input())+1):
    str1 = input()
    str2 = input()

    check_arr = [0] * 26 # str1 해당 글자가 있는지 체크
    count_arr = [0] * 26 # 해당 글자 카운트

    # str1을 순회하면서 알파벳 체크
    for i in str1:
        check_arr[ord(i)-ord('A')] = 1

    # 체크된 알파벳의 카운트 세기
    for i in str2:
        if check_arr[ord(i)-ord('A')]:
            count_arr[ord(i)-65] += 1
    print("#{} {}".format(tc, max(count_arr)))
'''