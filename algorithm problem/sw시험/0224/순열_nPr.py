# nPr
def perm(n, k):  # 원소의 개수, k:현재 뎁스
    if R == k:   # n -> R
        print(t)
    else:
        for i in range(n):
            # 방문했으면 건너 뛴다.
            if visited[i]: continue

            # 바꿔준다.
            t[k] = arr[i]

            # 방문 기록 남기기
            visited[i] = 1
            # 앞으로
            perm(n, k + 1)
            # 다시 뒤로 갈때
            visited[i] = 0


arr = [1, 2, 3]
N = len(arr)
R = 2
t = [0] * R  # 원소의 순서를 저장
visited = [0] * N
perm(N, 0)