# Recommender System
Implementing a Recommender system using Matrix Factorization Collaborative Filtering

In this project, our goal is to recommend top 5 movies to a user, based on Matrix Factorization, using MovieLens 20M dataset.
You can download the dataset from [kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset).
Four steps are taken through this project. Corresponding (.py) files should run in this order:

- Preprocess the data (preprocess.py)
- Data analysis (analyzie.py)
- Create model (learning.py)
- Predict user rating (predict.py)

## 1- Preprocess:

Since processing 20 million ratings takes a lot of time, we will use a subset of dataset.
So our first step is to shrink data into a reasonable amount by choosing most common user and movies.
Then, an id-correction is needed in order to fill dataset with identifiers starting from 0 to N-1.
Finally, we will shuffle the data and divide dataset into training and test data.
The result is shown as below:

![preprocess](https://user-images.githubusercontent.com/36487462/94345788-4e307f80-0035-11eb-91a6-5209563fb54a.jpg)

## 2- Data analysis:

A distribution of important data such as rating, movie genres and publication year of movies is plotted for better data understanding.


![year](https://user-images.githubusercontent.com/36487462/94345824-90f25780-0035-11eb-9c0a-b4daa822cfda.png)


![rating](https://user-images.githubusercontent.com/36487462/94345833-9e0f4680-0035-11eb-9390-f3df2793e922.png)


![genre](https://user-images.githubusercontent.com/36487462/94345847-b0898000-0035-11eb-87b7-1f9e2f9f0905.png)

## 3- Create model

In this section, model will be created. Later we will plot results of our loss function, which is Mean Squarred Error (MSE) in this project.
The model will be trained within 25 epochs.

![epoch](https://user-images.githubusercontent.com/36487462/94345890-f0e8fe00-0035-11eb-8dae-ae6c0e98dd9e.png)

After 25th epoch:

![MSE](https://user-images.githubusercontent.com/36487462/94345923-1bd35200-0036-11eb-8cea-4e896b174965.png)

## 4- Predict user rating

For a specific user, ratings over unseen movies will be generated. Then we will recommend top 5 movies that user might like.

![recommend](https://user-images.githubusercontent.com/36487462/94346028-c0ee2a80-0036-11eb-8839-f2b233de1150.png)

![user taste](https://user-images.githubusercontent.com/36487462/94346022-b6cc2c00-0036-11eb-8bd2-a52640e7808f.png)
