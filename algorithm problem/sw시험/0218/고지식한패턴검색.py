def brute_force(t, p):
    N = len(t)
    M = len(p)

    for i in range(N-M+1):  # N-M+1 p값의 길이를 빼준다.
        cnt = 0             # 맞춘 개수
        for j in range(M):
            if t[i+j] == p[j]:
                cnt += 1
            else:
                break
        if cnt == M:        # 맞춘 단어 길이가 M과 같으면 자리반환
            return i
    return -1 


t = "This is a book"
p = "is"
print(brute_force(t, p))