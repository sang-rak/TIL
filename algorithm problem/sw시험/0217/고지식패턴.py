t = "A patten matching algorithm"
p = "rithm"

M = len(p)
N = len(t)

# while 문 버전
def BruteForce(p, t):
    i = 0
    j = 0
    while j < M and i < N :
        if t[i] != p[j] :
            i = i - j
            j = -1 # 아니면 다시 보내기
        i += 1
        j += 1
    if j == M: return i - M # 검색 성공
    else: return -1 # 검색 실패

# for 문 버전
def BruteForce2(p, t):
    N = len(t)
    M = len(p)

    for i in range(N-M+1):
        cnt = 0
        for j in range(M):
            if t[i+j] == p[j]:
                cnt += 1
            else:
                break
        if cnt == M:
            return i
    return -1

print(BruteForce2(p, t))