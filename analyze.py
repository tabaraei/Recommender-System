import matplotlib.pyplot as plt
import pandas as pd


def histogram(data, xlabel='', hist=True, rotation=0):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.canvas.set_window_title(xlabel)
    ax.set_title('Distribution of ' + xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')
    plt.xticks(rotation=rotation)

    if hist:
        plt.hist(data, bins=len(set(data)), color='green',
                 alpha=0.5, ec='#666', linewidth=2)
    else:
        plt.bar(list(data.keys()), list(data.values()),
                color='green', alpha=0.5, ec='#666', linewidth=2)
    plt.show()


def plotYear(data):
    yearNotMentioned = 0
    years = []
    for title in data:
        publishYear = title[-5:-1]
        years.append(int(publishYear))
    histogram(years, xlabel='Movie Publish Year')


def plotRating(data):
    histogram(data, xlabel='Rating')


def plotGenre(data):
    genreList = {}
    for movieGenres in data:
        for genre in movieGenres.split('|'):
            if genre in genreList:
                genreList[genre] += 1
            else:
                genreList[genre] = 0
    histogram(genreList, xlabel='Genre', hist=False, rotation=60)


df = pd.read_csv('drive/My Drive/MovieLens 20M/df.csv')
plotYear(df.title)
plotRating(df.rating)
plotGenre(df.genres)
