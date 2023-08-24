#import libraries

import pandas as pd
import numpy as np
import warnings# warning filter
from collections import deque

#ML model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#warning handle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

#Read in movie data from netflix prize data
def manual_separation(bad_line):
    right_split = bad_line[:-2] + [",".join(bad_line[-2:])] # All the "bad lines" where all coming from the same last column that was containing ","
    return right_split
movie_titles = pd.read_csv("input/movie_titles.csv", encoding = "ISO-8859-1", header = None, names = ['Id', 'Year', 'Name'], on_bad_lines=manual_separation,
        engine="python")
movie_titles.set_index('Id', inplace = True)

#Read in a movie metadata dataset
movie_metadata = pd.read_csv('input/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
# Removes movies with low amount of ratings
movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)

#Read in User Data and Preprocess
df_raw = pd.read_csv('input/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])


# Find empty rows to slice dataframe for each movie
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)


# Gather all dataframes
user_data = []

# Iterate over all movies
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    
    # Check if it is the last movie in the file
    if df_id_1<df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()
        
    # Create movie_id column
    tmp_df['Movie'] = movie_id
    
    # Append dataframe to list
    user_data.append(tmp_df)

# Combine all dataframes and deletes temp variables
df = pd.concat(user_data)
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

#Clean and Filter Data
#Filters movies with low amount of ratings
min_movie_ratings = 8000
filter_movies = (df['Movie'].value_counts()>min_movie_ratings)
filter_movies = filter_movies[filter_movies].index.tolist()

# Filters users with low amount of ratings
min_user_ratings = 200
filter_users = (df['User'].value_counts()>min_user_ratings)
filter_users = filter_users[filter_users].index.tolist()

#APplying filters
df_filter = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]
del filter_movies, filter_users, min_movie_ratings, min_user_ratings

df_pivot = df_filter.pivot_table(index="User", columns="Movie", values="Rating")

# Fill in NaN values of pivot table
df_pivot_fill = df_pivot.T.fillna(df_pivot.mean(axis=1)).T

#Pearson
f = ["count", "mean"]

df_movie_summary = df.groupby("Movie")["Rating"].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)

def recommend(movie_title, min_count):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(movie_titles.index[movie_titles["Name"] == movie_title][0])
    target = df_pivot_fill[i]
    similar_to_target = df_pivot_fill.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns=["PearsonR"])
    corr_target.dropna(inplace=True)
    corr_target = corr_target.sort_values("PearsonR", ascending=False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(movie_titles).join(df_movie_summary)[
        ["PearsonR", "Name", "count", "mean"]
    ]
    print(corr_target[corr_target["count"] > min_count][:10].to_string(index=False))

#Cosine Similarity using TFIDF 

# Create tf-idf matrix for text comparison
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_metadata['overview'].dropna())


# Compute cosine similarity between all movie-descriptions
similarity = cosine_similarity(tfidf_matrix)
# Remove self-similarity from matrix
similarity -= np.eye(similarity.shape[0])


def cosine_recommend(movie_title):
    number = 1
    movie = movie_title
    n_plot = 10
    index = movie_metadata.reset_index(drop=True)[movie_metadata.index==movie].index[0]
    # Get indices and scores of similar movies
    similar_movies_index = np.argsort(similarity[index])[::-1][:n_plot]
    similar_movies_score = np.sort(similarity[index])[::-1][:n_plot]

    # Get titles of similar movies
    similar_movie_titles = movie_metadata.iloc[similar_movies_index].index
    print("Cosine Similarity using TFIDF Matrices")
    print("For movie ({})".format(movie))
    for i in range(n_plot):
        print(f'#{number}: Title:{similar_movie_titles[i]}, ID:{similar_movies_index[i]}, Score:{similar_movies_score[i]}')
        number +=1

