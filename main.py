import pandas as pd
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

pd.set_option("display.max_columns",None)

movies_dataset = pd.read_csv("data/tmdb_5000_movies.csv")
credits_dataset = pd.read_csv("data/tmdb_5000_credits.csv")
movies_dataset.head()
movies_dataset.shape
movies = movies_dataset.merge(credits_dataset, on="title")
movies.columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]  #choosing important columns

movies.isnull().sum()   #We can drop this missing values. Because there are a little missing value so there is no influence on the model
movies.dropna(inplace=True)
movies.duplicated().sum()
#We want to pull the name variable from the variable with convert func.
def convert(text):
    liste = []
    for i in ast.literal_eval(text):
        liste.append(i['name'])
    return liste

#We get the movie types in the genres variable
movies['genres'] = movies['genres'].apply(convert)
#we get the relevant keywords from keywords column.
movies['keywords'] = movies['keywords'].apply(convert)
#It was created to pull 3 name data from the variable.
def convertcharacter(text):
    liste = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter<3:
            liste.append(i['name'])
        counter+=1
    return liste
#We get 3 actor names from the cast variable
movies['cast'] = movies['cast'].apply(convertcharacter)
#If job is director, we pull the name.
def fetch_director(text):
    liste = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            liste.append(i['name'])
            break
    return liste
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
def remove_space(word):
    liste = []
    for i in word:
        liste.append(i.replace(" ",""))
    return liste

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_movie_df = movies[['movie_id','title','tags']]
new_movie_df = movies[['movie_id', 'title', 'tags']].copy()

new_movie_df["tags"].dtype
new_movie_df['tags'] = new_movie_df['tags'].apply(lambda x: " ".join(x))
new_movie_df.head(2)
ps = PorterStemmer()
def stems(text):
    liste = []
    for i in text.split():
        liste.append(ps.stem(i))
    return " ".join(liste)

new_movie_df['tags'] = new_movie_df['tags'].apply(stems)
new_movie_df.iloc[0]['tags']
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_movie_df['tags']).toarray()
# vector.shape
similarity = cosine_similarity(vector)
# similarity.shape
new_movie_df[new_movie_df['title'] == 'Spider-Man'].index[0]
def recommend(movie):
    index = new_movie_df[new_movie_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True, key=lambda x:x[1])
    for i in distances[1:6]:
        print(new_movie_df.iloc[i[0]].title)

recommend('Spider-Man')
recommend('The Dark Knight Rises')

pickle.dump(new_movie_df,open("data/movie_list.pkl", "wb"))
pickle.dump(similarity,open("data/similarity.pkl", "wb"))






