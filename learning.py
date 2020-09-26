import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

# load in the data
with open('drive/My Drive/MovieLens 20M/json/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('drive/My Drive/MovieLens 20M/json/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('drive/My Drive/MovieLens 20M/json/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('drive/My Drive/MovieLens 20M/json/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

# initialize number of users and movies
usersCount = len(user2movie)
moviesCount = len(movie2user)
print('Users Count:', usersCount)
print('Movies Count:', moviesCount)

# initialize variables
# k: latent factor
# W: user matrix,  b: user bias
# U: movie matrix, c: movie bias
# mu: mean of ratings
K = 10
W = np.random.randn(usersCount, K)
b = np.zeros(usersCount)
U = np.random.randn(moviesCount, K)
c = np.zeros(moviesCount)
mu = np.mean(list(usermovie2rating.values()))
trainLoss = []
testLoss = []
regularization = 0.02
epochs = 25

# calculate loss on test or training data
# data: (userId, movieId) -> rating


def MeanSquaredError(data):
    N = float(len(data))
    sumSquareError = 0
    for usermovie, rating in data.items():
        user, movie = usermovie
        predictedRating = W[user].dot(U[movie]) + b[user] + c[movie] + mu
        # MSE (mean squared errors): 1/n sigma (r-r')^2
        # RMSE (root mean squared error): sqrt(MSE)
        # MAE (mean absolute error): 1/n sigma |r-r'|
        sumSquareError += (predictedRating - rating)*(predictedRating - rating)
    return sumSquareError / N


# learning algorithm
for epoch in range(epochs):
    print('---------------------------------')
    print('epoch:', epoch, '\n')
    epoch_start = datetime.now()
    # perform updates

    # ---------------------------------------------------
    # update W and b (user)
    t0 = datetime.now()
    for i in range(usersCount):
        # for W
        matrix = np.eye(K) * regularization
        vector = np.zeros(K)

        # for b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i, j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu)*U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)

        # set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + regularization)

        if i % (usersCount//10) == 0:
            print("i:", i, "usersCount:", usersCount)
    print("updated W and b:", datetime.now() - t0, '\n')

    # ---------------------------------------------------
    # update U and c (movie)
    t0 = datetime.now()
    for j in range(moviesCount):
        # for U
        matrix = np.eye(K) * regularization
        vector = np.zeros(K)

        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu)*W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)

            # set the updates
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + regularization)

            if j % (moviesCount//10) == 0:
                print("j:", j, "moviesCount:", moviesCount)
        except KeyError:
            # possible not to have any ratings for a movie
            pass
    print("updated U and c:", datetime.now() - t0, '\n')
    print("epoch duration:", datetime.now() - epoch_start)

    # ---------------------------------------------------
    # store train loss
    t0 = datetime.now()
    trainLoss.append(MeanSquaredError(usermovie2rating))

    # store test loss
    testLoss.append(MeanSquaredError(usermovie2rating_test))
    print('Calculate cost:', datetime.now() - t0)
    print('Train Loss:', trainLoss[-1])
    print('Test Loss:', testLoss[-1])


print('\nTrain Losses:', trainLoss)
print('Test Losses:', testLoss)

# store model
with open('drive/My Drive/MovieLens 20M/model/W.npy', 'wb') as f:
    np.save(f, W)
with open('drive/My Drive/MovieLens 20M/model/U.npy', 'wb') as f:
    np.save(f, U)
with open('drive/My Drive/MovieLens 20M/model/b.npy', 'wb') as f:
    np.save(f, b)
with open('drive/My Drive/MovieLens 20M/model/c.npy', 'wb') as f:
    np.save(f, c)
with open('drive/My Drive/MovieLens 20M/model/mu.npy', 'wb') as f:
    np.save(f, mu)

# plot losses
plt.plot(trainLoss, label='Train Loss')
plt.plot(testLoss, label='Test Loss')
plt.title('Mean Squared Error')
plt.legend()
plt.show()
