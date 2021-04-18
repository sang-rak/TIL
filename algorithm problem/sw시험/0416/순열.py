def f(i, n, k):  # p[i]를 정하는 함수
    if i == k:
        print(p)
    else:
        for j in range(n):
            if u[j] == 0:     # j를 쓴적이 없으면
                u[j] = 1      # 썼다고 기록하고
                p[i] = j      # p[i]의 숫자로 정함
                f(i+1, n, k)  # p[i+1] 숫자를 정하로 이동
                u[j] = 0      # j를 다른 자리에 쓸 수 있도록 함


N = 5
K = 3

p = [0] * K
u = [0] * N
f(0, N, K)