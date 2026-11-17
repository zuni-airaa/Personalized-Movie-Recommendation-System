# ============================================================
# PERSONALIZED MOVIE RECOMMENDATION AGENT USING Q-LEARNING
# Advanced Machine Learning Mini Project
# ============================================================

import os
import zipfile
import urllib.request
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = os.path.join(DATA_FOLDER, "ml-1m.zip")
EXTRACT_FOLDER = os.path.join(DATA_FOLDER, "ml-1m")

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODES = 30
TOP_K = 10

random.seed(42)
np.random.seed(42)


# ============================================================
# 2. DOWNLOAD DATASET
# ============================================================

def download_dataset():

    ratings_file = os.path.join(EXTRACT_FOLDER, "ratings.dat")

    if not os.path.exists(ratings_file):

        print("Downloading MovieLens 1M dataset...")

        if not os.path.exists(ZIP_PATH):
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

        print("Extracting dataset...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
            zip_file.extractall(DATA_FOLDER)

    print("Dataset downloaded successfully!")


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():

    ratings = pd.read_csv(
        os.path.join(EXTRACT_FOLDER, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )

    movies = pd.read_csv(
        os.path.join(EXTRACT_FOLDER, "movies.dat"),
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    print("\nDataset Information")
    print("-------------------")
    print("Total Ratings:", len(ratings))
    print("Total Users:", ratings["UserID"].nunique())
    print("Total Movies:", movies["MovieID"].nunique())

    return ratings, movies


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(ratings, movies):

    print("\nRating Distribution:")
    print(ratings["Rating"].value_counts().sort_index())

    print("\nMissing Values:")
    print(ratings.isnull().sum())

    print("\nDuplicate Ratings:", ratings.duplicated(
        ["UserID", "MovieID"]
    ).sum())

    # Rating distribution
    plt.figure(figsize=(8, 5))

    ratings["Rating"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Movie Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "rating_distribution.png")
    )

    plt.close()

    # Most popular movies
    popular = ratings.groupby("MovieID").size().sort_values(
        ascending=False
    ).head(10)

    popular = popular.reset_index(name="Rating Count")
    popular = popular.merge(movies, on="MovieID")

    plt.figure(figsize=(10, 6))

    plt.barh(
        popular["Title"],
        popular["Rating Count"]
    )

    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "popular_movies.png")
    )

    plt.close()

    print("\nEDA completed!")


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

def preprocess_data(ratings):

    ratings = ratings.drop_duplicates(
        subset=["UserID", "MovieID"]
    )

    ratings = ratings.sort_values("Timestamp")

    train_data = []
    test_data = []

    for user_id, group in ratings.groupby("UserID"):

        if len(group) < 5:
            train_data.append(group)
            continue

        split = int(len(group) * 0.8)

        train_data.append(group.iloc[:split])
        test_data.append(group.iloc[split:])

    train_data = pd.concat(train_data)

    if test_data:
        test_data = pd.concat(test_data)
    else:
        test_data = pd.DataFrame(columns=ratings.columns)

    print("\nData preprocessing completed!")

    return train_data, test_data


# ============================================================
# 6. Q-LEARNING ENVIRONMENT
# ============================================================

class MovieRecommendationAgent:

    def __init__(self, train_data, movies):

        self.train_data = train_data
        self.movies = movies.reset_index(drop=True)

        self.movie_ids = self.movies["MovieID"].tolist()

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.index_to_movie = {
            index: movie_id
            for movie_id, index in self.movie_to_index.items()
        }

        self.genres = sorted(
            set(
                genre
                for genre_list in self.movies["Genres"]
                for genre in genre_list.split("|")
            )
        )

        self.genre_to_index = {
            genre: index
            for index, genre in enumerate(self.genres)
        }

        self.num_states = len(self.genres) + 1
        self.unknown_state = len(self.genres)

        self.num_actions = len(self.movies)

        # Initialize Q-table
        self.Q = np.zeros(
            (self.num_states, self.num_actions)
        )

        self.movie_genres = {}

        for _, row in self.movies.iterrows():

            encoded_genres = [
                self.genre_to_index[g]
                for g in row["Genres"].split("|")
                if g in self.genre_to_index
            ]

            self.movie_genres[row["MovieID"]] = encoded_genres

        self.user_history = {}

        for user_id, group in train_data.groupby("UserID"):

            self.user_history[user_id] = group[
                ["MovieID", "Rating"]
            ].values.tolist()

        self.user_states = {}

        for user_id, history in self.user_history.items():

            self.user_states[user_id] = self.get_user_state(
                history
            )

    # --------------------------------------------------------
    # USER STATE
    # --------------------------------------------------------

    def get_user_state(self, history):

        genre_scores = np.zeros(len(self.genres))

        for movie_id, rating in history:

            movie_id = int(movie_id)

            for genre in self.movie_genres.get(movie_id, []):

                genre_scores[genre] += float(rating)

        if np.sum(genre_scores) == 0:
            return self.unknown_state

        return int(np.argmax(genre_scores))

    # --------------------------------------------------------
    # REWARD
    # --------------------------------------------------------

    def calculate_reward(self, rating):

        return (float(rating) - 3) / 2

    # --------------------------------------------------------
    # EPSILON-GREEDY ACTION SELECTION
    # --------------------------------------------------------

    def choose_action(self, state, available_actions, epsilon):

        if random.random() < epsilon:

            return random.choice(available_actions)

        q_values = self.Q[state, available_actions]

        best_position = np.argmax(q_values)

        return available_actions[best_position]

    # --------------------------------------------------------
    # Q-LEARNING UPDATE
    # --------------------------------------------------------

    def update_q_value(
        self,
        state,
        action,
        reward,
        next_state
    ):

        old_value = self.Q[state, action]

        future_value = np.max(self.Q[next_state])

        self.Q[state, action] = old_value + ALPHA * (
            reward + GAMMA * future_value - old_value
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(self):

        epsilon = EPSILON

        rewards_per_episode = []

        print("\nStarting Q-Learning Training...\n")

        for episode in range(EPISODES):

            episode_rewards = []

            for user_id, history in self.user_history.items():

                if len(history) < 2:
                    continue

                state = self.user_states[user_id]

                # Use the movies rated by this user
                # as available actions during offline training.
                available_actions = [
                    self.movie_to_index[int(movie_id)]
                    for movie_id, rating in history
                    if int(movie_id) in self.movie_to_index
                ]

                if not available_actions:
                    continue

                # Limit training steps
                selected_actions = random.sample(
                    available_actions,
                    min(20, len(available_actions))
                )

                for action in selected_actions:

                    action_movie = self.index_to_movie[action]

                    rating = next(
                        float(rating)
                        for movie_id, rating in history
                        if int(movie_id) == action_movie
                    )

                    reward = self.calculate_reward(rating)

                    next_state = state

                    self.update_q_value(
                        state,
                        action,
                        reward,
                        next_state
                    )

                    episode_rewards.append(reward)

            epsilon = max(
                EPSILON_MIN,
                epsilon * EPSILON_DECAY
            )

            average_reward = (
                np.mean(episode_rewards)
                if episode_rewards else 0
            )

            rewards_per_episode.append(average_reward)

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Reward: {average_reward:.4f} | "
                f"Epsilon: {epsilon:.4f}"
            )

        np.save(
            os.path.join(OUTPUT_FOLDER, "q_table.npy"),
            self.Q
        )

        print("\nQ-Learning Training Completed!")

        return rewards_per_episode

    # --------------------------------------------------------
    # MOVIE RECOMMENDATIONS
    # --------------------------------------------------------

    def recommend(self, user_id, top_k=10):

        state = self.user_states.get(
            user_id,
            self.unknown_state
        )

        watched = {
            int(movie_id)
            for movie_id, rating
            in self.user_history.get(user_id, [])
        }

        q_values = self.Q[state].copy()

        for index, movie_id in self.index_to_movie.items():

            if movie_id in watched:
                q_values[index] = -np.inf

        best_actions = np.argsort(q_values)[::-1][:top_k]

        return [
            self.index_to_movie[action]
            for action in best_actions
            if np.isfinite(q_values[action])
        ]


# ============================================================
# 7. POPULARITY BASELINE
# ============================================================

def popularity_recommendations(
    train_data,
    user_id,
    top_k=10
):

    watched = set(
        train_data[
            train_data["UserID"] == user_id
        ]["MovieID"]
    )

    popularity = train_data.groupby(
        "MovieID"
    )["Rating"].count().sort_values(ascending=False)

    recommendations = [
        movie_id
        for movie_id in popularity.index
        if movie_id not in watched
    ]

    return recommendations[:top_k]


# ============================================================
# 8. SIMPLIFIED FITTED Q-LEARNING
# ============================================================

def fitted_q_learning(agent):

    print("\nTraining Simplified Fitted Q-Learning...")

    X = []
    y = []

    for user_id, history in agent.user_history.items():

        state = agent.user_states[user_id]

        for movie_id, rating in history:

            movie_id = int(movie_id)

            if movie_id not in agent.movie_to_index:
                continue

            action = agent.movie_to_index[movie_id]

            reward = agent.calculate_reward(rating)

            target = (
                reward + GAMMA * np.max(agent.Q[state])
            )

            X.append([state, action])
            y.append(target)

    if not X:
        return None

    X = np.array(X)
    y = np.array(y)

    # Limit the number of samples
    if len(X) > 80000:

        selected = np.random.choice(
            len(X),
            80000,
            replace=False
        )

        X = X[selected]
        y = y[selected]

    model = ExtraTreesRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print("Fitted Q-Learning completed!")

    return model


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(
    recommendations,
    test_data,
    movies
):

    hit_rates = []
    precisions = []
    recalls = []

    all_recommended = set()

    for user_id, group in test_data.groupby("UserID"):

        relevant_movies = set(
            group[group["Rating"] >= 4]["MovieID"]
        )

        if not relevant_movies:
            continue

        user_recommendations = recommendations.get(
            user_id, []
        )[:TOP_K]

        all_recommended.update(user_recommendations)

        hits = (
            set(user_recommendations) & relevant_movies
        )

        hit_rates.append(
            1 if hits else 0
        )

        precisions.append(
            len(hits) / TOP_K
        )

        recalls.append(
            len(hits) / len(relevant_movies)
        )

    return {
        "Hit Rate@10": np.mean(hit_rates) if hit_rates else 0,
        "Precision@10": np.mean(precisions) if precisions else 0,
        "Recall@10": np.mean(recalls) if recalls else 0,
        "Coverage": len(all_recommended) / len(movies)
    }


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():

    print("\n==============================================")
    print("PERSONALIZED MOVIE RECOMMENDATION USING Q-LEARNING")
    print("==============================================")

    # Download and load data
    download_dataset()

    ratings, movies = load_dataset()

    # EDA
    perform_eda(ratings, movies)

    # Preprocessing
    train_data, test_data = preprocess_data(ratings)

    # Initialize agent
    agent = MovieRecommendationAgent(
        train_data,
        movies
    )

    # Train Q-learning
    training_rewards = agent.train()

    # Plot training rewards
    plt.figure(figsize=(8, 5))

    plt.plot(training_rewards)

    plt.title("Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "training_rewards.png")
    )

    plt.close()

    # Generate Q-learning recommendations
    q_recommendations = {}

    for user_id in test_data["UserID"].unique():

        q_recommendations[user_id] = agent.recommend(
            user_id,
            TOP_K
        )

    # Generate popularity recommendations
    popularity_results = {}

    for user_id in test_data["UserID"].unique():

        popularity_results[user_id] = (
            popularity_recommendations(
                train_data,
                user_id,
                TOP_K
            )
        )

    # Train fitted Q-learning model
    fitted_model = fitted_q_learning(agent)

    # Evaluate models
    q_metrics = evaluate_model(
        q_recommendations,
        test_data,
        movies
    )

    popularity_metrics = evaluate_model(
        popularity_results,
        test_data,
        movies
    )

    results = [
        {
            "Model": "Q-Learning",
            **q_metrics
        },
        {
            "Model": "Popularity Baseline",
            **popularity_metrics
        }
    ]

    # Fitted Q-learning recommendations
    if fitted_model is not None:

        fitted_recommendations = {}

        for user_id in test_data["UserID"].unique():

            state = agent.user_states.get(
                user_id,
                agent.unknown_state
            )

            watched = {
                int(movie_id)
                for movie_id, rating
                in agent.user_history.get(user_id, [])
            }

            candidate_actions = [
                index
                for index, movie_id
                in agent.index_to_movie.items()
                if movie_id not in watched
            ]

            if candidate_actions:

                X_test = np.array([
                    [state, action]
                    for action in candidate_actions
                ])

                predicted_values = fitted_model.predict(X_test)

                best_positions = np.argsort(
                    predicted_values
                )[::-1][:TOP_K]

                fitted_recommendations[user_id] = [
                    agent.index_to_movie[
                        candidate_actions[position]
                    ]
                    for position in best_positions
                ]

            else:
                fitted_recommendations[user_id] = []

        fitted_metrics = evaluate_model(
            fitted_recommendations,
            test_data,
            movies
        )

        results.append({
            "Model": "Simplified Fitted Q-Learning",
            **fitted_metrics
        })

    # Display results
    results_df = pd.DataFrame(results)

    print("\n========== MODEL EVALUATION ==========")

    print(results_df.to_string(index=False))

    # Save evaluation results
    results_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "evaluation_results.csv"),
        index=False
    )

    # Model comparison graph
    results_df.set_index("Model").plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Recommendation Model Comparison")
    plt.ylabel("Evaluation Score")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "model_comparison.png")
    )

    plt.close()

    # Save movie information
    movies.to_csv(
        os.path.join(OUTPUT_FOLDER, "movie_mapping.csv"),
        index=False
    )

    # Display sample recommendations
    if len(test_data) > 0:

        sample_user = int(test_data["UserID"].iloc[0])

        print("\n========== SAMPLE RECOMMENDATIONS ==========")

        print("User ID:", sample_user)

        for movie_id in q_recommendations.get(sample_user, []):

            title = movies[
                movies["MovieID"] == movie_id
            ]["Title"].values

            if len(title) > 0:
                print("-", title[0])

    print("\nProject execution completed!")
    print("Check the outputs folder for results and graphs.")


# ============================================================
# 11. RUN PROJECT
# ============================================================

if __name__ == "__main__":
    main()