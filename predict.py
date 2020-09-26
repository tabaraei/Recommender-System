import pandas as pd
import numpy as np

df = pd.read_csv('drive/My Drive/MovieLens 20M/df.csv')

with open('drive/My Drive/MovieLens 20M/model/W.npy', 'rb') as f:
    W = np.load(f)
with open('drive/My Drive/MovieLens 20M/model/U.npy', 'rb') as f:
    U = np.load(f)
with open('drive/My Drive/MovieLens 20M/model/b.npy', 'rb') as f:
    b = np.load(f)
with open('drive/My Drive/MovieLens 20M/model/c.npy', 'rb') as f:
    c = np.load(f)
with open('drive/My Drive/MovieLens 20M/model/mu.npy', 'rb') as f:
    mu = np.load(f)

print('Dimension of W: ', W.shape)
print('Dimension of U: ', U.shape, '\n')

# predict for user 3
user = 3
predictedMovies = []

# looking among all movies
for i in range(U.shape[0]):
    if i not in list(df.movieId[df.userId == user]):
        # among movies that user haven't seen yet
        predict = W[user].dot(U[i]) + b[user] + c[i] + mu
        print('Prediction of movie ' + str(i) + ' for user ' +
              str(user) + ' is: ' + str(predict))
        predictedMovies.append([predict, i])

# sort by highest ratings
predictedMovies.sort(reverse=True)

print('\nTop 5 movies to recommend are:')
for rating, id in predictedMovies[:5]:
    rowIndex = df.index[df.movieId == id]
    title = df.iloc[rowIndex[0]].title
    genre = df.iloc[rowIndex[0]].genres
    print('---------------------------------')
    print('Title:', title)
    print('Genre:', genre)

print('\nRating of user', user, 'over other movies:')
df[df.userId == user]
