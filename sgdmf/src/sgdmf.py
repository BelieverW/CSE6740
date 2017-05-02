import numpy
import pandas as pd
from math import sqrt
from random import randint

def matrix_factorization(R, testR, P, Q, K, beta=0.045, steps=5000, alpha=0.0001):
    Q = Q.T
    preE = 1
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        #eR = numpy.dot(P,Q)
        e = 0
        cnt = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    cnt += 1
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
        e = sqrt(e / cnt)
        append_line = str(e) + "\n"
        with open("../res_output/iteration_train.txt", "a") as myfile:
            myfile.write(append_line)

        test_rmse = 0
        cnt = 0
        for i in range(len(testR)):
            for j in range(len(testR[i])):
                if testR[i][j] > 0:
                    cnt += 1
                    test_rmse = test_rmse + pow(testR[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
        test_rmse = sqrt(test_rmse / cnt)
        with open("../res_output/iteration_test.txt", "a") as myfile:
            myfile.write(str(test_rmse) + "\n")
        if abs(preE - e) / preE < 0.0001:
            break
        else:
            preE = e

    test_rmse = 0
    cnt = 0
    for i in range(len(testR)):
        for j in range(len(testR[i])):
            if testR[i][j] > 0:
                cnt += 1
                test_rmse = test_rmse + pow(testR[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
    test_rmse = sqrt(test_rmse / cnt)
    return test_rmse

###############################################################################


if __name__ == "__main__":


    ratings_list = [i.strip().split(",") for i in open('../small_dataset/ratings.csv', 'r', encoding='utf-8').readlines()]

    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)

    R_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    #print(R_df.head())

    R = R_df.as_matrix()
    x = numpy.array(R, dtype='|S4')
    RR = x.astype(numpy.float)
    print(RR.shape)


    N = len(RR)
    M = len(RR[0])
    K = 11

    TrainData = numpy.zeros((N, M))
    TestData = numpy.zeros((N, M))

    for i in range(N):
        for j in range(M):
            if R[i][j] != 0 and randint(0, 9) < 8:
                TrainData[i][j] = R[i][j]
            else:
                TestData[i][j] = R[i][j]

    # numpy.savetxt('TrainDate.csv', TrainData, delimiter=',')
    # numpy.savetxt('TestData.csv', TestData, delimiter=',')
    P = numpy.random.rand(N, K)
    Q = numpy.random.rand(M, K)
    test_rmse = matrix_factorization(TrainData, TestData, P, Q, K)

