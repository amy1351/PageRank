import numpy as np

#helper function to sort the indexed pagerank vector
#return eigenvector element
def sortFirst(val):
    return val[0]

#helper function to compute initial nxn probability matrix P
#P[i][j] is 1/L(j) if there is a link from j to i, 0 otherwise
def computeP(links):
    n = len(links)
    #initialize P with all 0's
    P = [[0 for i in range(n)] for j in range(n)]
    #P[i][j] is 1/L(j) if there is a link from j to i
    for j in range(len(links)):
        numLinks = len(links[j])
        for i in links[j]:
            P[i][j] = 1/numLinks
    return P

#Google PageRank algorithm, ranks pages by order of importance
#input a Markov chain as a 2D array, output a set of rankings as a 1D array
def rank(links):
    n = len(links)
    #compute initial nxn probability matrix
    P = computeP(links)

    #compute modified matrix PB with damping factor
    #PB = (1-B)P + B(Q)
    #initialize Q with 1 / number of nodes, to account for dangling nodes
    Q = [[(1/n) for i in range(n)] for j in range(n)]
    #set damping factor B to Google's damping factor, 0.15
    B = 0.15
    #loop through Q to compute B*Q
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            Q[i][j] = B * Q[i][j]
    #loop through P to compute (1-B)P
    for i in range(len(P)):
        for j in range(len(P[i])):
            P[i][j] = (1-B) * P[i][j]
    PB = np.add(P, Q)

    #compute eigenvalues and eigenvectors of PB using numpy
    #w is a 1D array with eigenvalues
    #v is a 2D array with columns as eigenvectors corresponding to eigenvalues in w
    w, v = np.linalg.eig(PB)

    #loop through w to find index of largest eigenvalue
    #note: we know the largest eigenvalue is 1 by Perron-Frobenis Theorem
    largestIndex = 0
    largest = 0
    for i in range(len(w)):
        if (w[i] > largest):
            largest = w[i]
            largestIndex = i
    #find eigenvector corresponding to index of largest eigenvalue
    pagerankVector = v[:,largestIndex]

    #compute page rank from final eigenvector
    #attach page index to each element in final eigenvector
    indexedPagerankVector = []
    for i in range(len(pagerankVector)):
        indexedPagerankVector.append([pagerankVector[i], i])
    #sort indexed pagerank vector by element
    indexedPagerankVector.sort(key = sortFirst, reverse = True)
    #obtain pagerank using index of the indexed final vector,
    #which is the 2nd element of the tuple
    pagerank = []
    for i in range(len(indexedPagerankVector)):
        pagerank.append(indexedPagerankVector[i][1])
    return pagerank

#test cases
links1 = [[1, 2, 3], [3], [0, 3], [0, 2]]
links2 = [[1, 2, 3], [4, 3], [0, 3], [6, 1], [6], [4, 7], [5], [5, 6]]
links3 = [[], [0, 2, 3, 4], [1, 4], [4], [5], [3, 6], [4, 5]]
links4 = [[1, 5], [2, 5], [1, 3, 5], [4], [1, 5], [2, 6], [0, 1]]
links5 = [[1,3,4],[0,2,4],[3,6],[2,4,6],[5,8],[4,6,8],[0,7,9],[0,6,8],[2,9],[0,2,8]]
print(rank(links5))
