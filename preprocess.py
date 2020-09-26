import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
import pickle

rating = pd.read_csv('drive/My Drive/MovieLens 20M/rating.csv')
movies = pd.read_csv('drive/My Drive/MovieLens 20M/movie.csv')
df = pd.merge(left=rating, right=movies, on='movieId')

print('Original dataframe:')
print('Number of ratings: ', len(df))
print('Number of unique users: ', df.userId.nunique())
print('Number of unique movies: ', df.movieId.nunique(), '\n')

# count of each user or movie occurred
usersHist = Counter(df.userId)
moviesHist = Counter(df.movieId)

# number of users and movies we would like to keep
n = 20000
m = 4000

# get array of most common users and movies, and shuffle the data
chosenUserIds = shuffle(
    [chosenUsers for chosenUsers, count in usersHist.most_common(n)])
chosenMovieIds = shuffle(
    [chosenMovies for chosenMovies, count in moviesHist.most_common(m)])
df = df[df.userId.isin(chosenUserIds) & df.movieId.isin(chosenMovieIds)].copy()

# Correct userId range
mapToNewUserId = {}
newId = 0
for oldId in chosenUserIds:
    mapToNewUserId[oldId] = newId
    newId += 1

# Correct movieId range
mapToNewMovieId = {}
newId = 0
for oldId in chosenMovieIds:
    mapToNewMovieId[oldId] = newId
    newId += 1

# assign newIDs to old dataframe
df.userId = df.apply(lambda row: mapToNewUserId[row.userId], axis=1)
df.movieId = df.apply(lambda row: mapToNewMovieId[row.movieId], axis=1)
df.drop(columns=['timestamp'], inplace=True)
df.sort_values(['userId', 'movieId'], inplace=True)
df = df[['userId', 'movieId', 'title', 'genres', 'rating']]
df.to_csv('drive/My Drive/MovieLens 20M/df.csv', index=False)

print('Shrinked dataframe:')
print('Number of ratings:', len(df))
print('Number of unique users:', df.userId.nunique())
print('Number of unique movies:', df.movieId.nunique(), '\n')

# split data into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
dfTrain = df.iloc[:cutoff]
dfTest = df.iloc[cutoff:]

# a dictionary to tell us which users have rated which movies
user2movie = {}
# a dicationary to tell us which movies have been rated by which users
movie2user = {}
# a dictionary to look up ratings in train data
usermovie2rating = {}
# a dictionary to look up ratings in test data
usermovie2rating_test = {}

# Create training data
print('Create training and test data in progress:')
currentRow = 0


def createData(row, mode):
    user = int(row.userId)
    movie = int(row.movieId)

    if mode == 'train':
        # {'user': [movie1, movie2, ..]}
        if user not in user2movie:
            user2movie[user] = [movie]
        else:
            user2movie[user].append(movie)
        # {'movie': [user1, user2, ..]}
        if movie not in movie2user:
            movie2user[movie] = [user]
        else:
            movie2user[movie].append(user)
        # {'(user,movie)': rating}
        usermovie2rating[(user, movie)] = row.rating
    else:
        usermovie2rating_test[(user, movie)] = row.rating

    global currentRow
    currentRow += 1
    if currentRow % 50000 == 0:
        print(round((currentRow/len(df))*100, 1), '%')


dfTrain.apply(lambda row: createData(row, 'train'), axis=1)
dfTest.apply(lambda row: createData(row, 'test'), axis=1)

# Save train and test data as JSON files
with open('drive/My Drive/MovieLens 20M/json/user2movie.json', 'wb') as f:
    pickle.dump(user2movie, f)

with open('drive/My Drive/MovieLens 20M/json/movie2user.json', 'wb') as f:
    pickle.dump(movie2user, f)

with open('drive/My Drive/MovieLens 20M/json/usermovie2rating.json', 'wb') as f:
    pickle.dump(usermovie2rating, f)

with open('drive/My Drive/MovieLens 20M/json/usermovie2rating_test.json', 'wb') as f:
    pickle.dump(usermovie2rating_test, f)

print('Training and test data saved successfully\n')
df = pd.read_csv('drive/My Drive/MovieLens 20M/df.csv')
print(df)
