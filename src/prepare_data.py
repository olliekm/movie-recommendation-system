import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_raw():
    """ Loads raw ratings and movies data from CSV files.
    """
    
    ratings = pd.read_csv('data/raw/ratings.csv')
    movies = pd.read_csv('data/raw/movies.csv')
    return ratings, movies

def build_movie_feature(movies: pd.DataFrame) -> pd.DataFrame:
    """ Builds data frame with one-hot encoded genres from the movies DataFrame.
    """

    # One hot encode genres
    movies["genre_list"] = movies["genres"].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()

    # Transform the genre list into a one-hot encoded DataFrame
    genre_mat = mlb.fit_transform(movies["genre_list"])

    # Create a DataFrame with the one-hot encoded genres
    genre_df = pd.DataFrame(genre_mat, columns=mlb.classes_, index=movies.index)

    # Concatenate the original movies DataFrame with the one-hot encoded genres
    movies = pd.concat([movies.drop("genres",axis=1), genre_df], axis=1)
    return movies

def join_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    # Merge ratings with movies on movieId
    merged_df = pd.merge(ratings, movies, on='movieId', how='left')
    
    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['timestamp', 'title'])
    
    return merged_df

def main():
    ratings, movies = load_raw()
    movies = build_movie_feature(movies)
    merged_df = join_features(ratings, movies)

    # Save the processed data to a CSV file
    merged_df.to_parquet('data/processed/ratings_movies.parquet', index=False)
    print("Data processing complete. Processed data saved to 'data/processed/ratings_movies.parquet'.")

if __name__ == "__main__":
    main()
