import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if       given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''


    def fit(self, reviews_pth='train_data.csv', movies_pth= 'movies_clean.csv', learning_rate=.01, iters=1):
        '''
        fit the recommender        to your dataset and also have this save the results
        pull from when you need to make predictions
        '''
        
        # Read in the datasets
        self.movies = pd.read_csv(movies_pth)
        self.reviews = pd.read_csv(reviews_pth)

        # define SVD parameters
        self.learning_rate = learning_rate
        self.iters = iters
        self.latent_features = 4
        
        # Create user-item matrix
        usr_itm = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = usr_itm.groupby(['user_id','movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_item_df)
        
        # find number of users and movies
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        
        # initialize the user and movie matrices with random values
        self.user_mat = np.random.rand(self.n_users, self.latent_features)
        self.movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0
        self.num_ratings = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(self.user_mat[i, :], self.movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2
                        
                        # update ratings counter
                        self.num_ratings += 1

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            self.user_mat[i, k] += self.learning_rate * (2*diff*self.movie_mat[k, j])
                            self.movie_mat[k, j] += self.learning_rate * (2*diff*self.user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
            

    def predict_rating(self, user_matrix, movie_matrix, user_id, movie_id):
        '''
        INPUT:
        user_matrix - user by latent factor matrix
        movie_matrix - latent factor by movie matrix
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        
        # Create series of users and movies in the right order
        user_ids_series = np.array(train_data_df.index)
        movie_ids_series = np.array(train_data_df.columns)

        # User row and Movie Column
        user_row = np.where(user_ids_series == user_id)[0][0]
        movie_col = np.where(movie_ids_series == movie_id)[0][0]

        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])

        return pred

    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = self.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = self.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(self.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names
    
    
    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        train_data - dataframe of data as user-movie matrix
        train_df - dataframe of training data reviews
        movies - movies df
        rec_num - number of recommendations to return (int)
        user_mat - the U matrix of matrix factorization
        movie_mat - the V matrix of matrix factorization

        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id                  
        rec_names - (array) a list or numpy array of recommended movies by name
        '''

        val_users = train_data_df.index
        rec_ids = create_ranked_df(movies,train_df)

        if _id_type == "user":
            idx = np.where(val_users == _id)[0][0]
            pred = predict_rating(self.user_mat, self.movie_mat, user_id, movie_id)
            preds = np.dot(user_mat[idx,:], self.movie_mat)
            indices = preds.argsort()[-rec_num][::-1]
            rec_ids = train_data_df.columns[indices]
            rec_names = get_movie_names[rec_ids]
        elif _id_type == "movie":
            rec_ids = find_similar_movies(_id)
            rec_names = get_movie_names(rec_ids)

        else:
            print("id is either a user_id nor a movie_id.")

        return rec_ids, rec_names
    
    def find_similar_movies(self, movie_id):
        '''
        INPUT
        movie_id - a movie_id 
        OUTPUT
        similar_movies - an array of the most similar movies by title
        '''
        # find the row of each movie id
        movie_idx = np.where(movies['movie_id'] == movie_id)[0][0]

        # find the most similar movie indices - to start I said they need to be the same for all content
        similar_idxs = np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]

        # pull the movie titles based on the indices
        similar_movies = np.array(movies.iloc[similar_idxs, ]['movie_id'])

        return similar_movies
    
    
    def get_movie_names(self, movie_ids):
        '''
        INPUT
        movie_ids - a list of movie_ids
        OUTPUT
        movies - a list of movie names associated with the movie_ids

        '''
        movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])

        return movie_lst
    

    def create_ranked_df(self, movies, reviews):
        '''
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe
        
        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, 
                        then time, and must have more than 4 ratings
        '''
        
        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]
        
        return ranked_movies
    

    def popular_recommendations(self, user_id, n_top, ranked_movies):
        '''
        INPUT:
        user_id - the user_id (str) of the individual you are making recommendations for
        n_top - an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

        OUTPUT:
        top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''

        top_movies = list(ranked_movies['movie'][:n_top])

        return top_movies
    
    

    

if __name__ == '__main__':
    # test different parts to make sure it works
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_pth='train_data.csv', movies_pth= 'movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # user in the dataset
    print(rec.make_recommendations(1,'user')) # user not in dataset
    print(rec.make_recommendations(1853728)) # movie in the dataset
    print(rec.make_recommendations(1)) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)

    
