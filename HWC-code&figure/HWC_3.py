def recSS(A, i, n):
    if i < n-1:
        small = i 
        for j in range(i+1, n):
            if A[j] < A[small]:
                small = j
        temp = A[small]
        A[small] = A[i]
        A[i] = temp
        print(A,i)
        i = i+1
        recSS(A, i, n)
        

A = [10, 13, 4, 7 ,11]
recSS(A, 0, 5)