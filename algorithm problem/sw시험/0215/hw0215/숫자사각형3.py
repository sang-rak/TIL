
'''
3
1 1
2 3
4 5

'''

T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T+1):
    # ///////////////////////////////////////////////////////////////////////////////////

    dr, dl = map(int, input().split())
    print(f'#{test_case}', end='\n')
    for i in range(1,dr+1):
        if i % 2 == 0:
            for j in range(dl-1,-1,-1):
                print(j+(i-1)*dl+1, end=' ')
            print('')
        else:
            for j in range(dl):
                print(j+(i-1)*dl+1, end=' ')
            print('')