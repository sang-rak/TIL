
'''
3
1
3
5

'''

T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T+1):
    # ///////////////////////////////////////////////////////////////////////////////////

    dr = int(input())
    print(f'#{test_case}', end='\n')
    for i in range(1, dr+1):

        for j in range(1,dr+1):
            print(i*j, end=' ')
        print('')