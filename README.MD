# Personalized Movie Recommendation Agent Using Q-Learning

## Project Description

This project implements a personalized movie recommendation system using Reinforcement Learning, specifically the Q-Learning algorithm.

The system learns user preferences from historical movie ratings and recommends movies based on those preferences.

## Objectives

- Implement Q-Learning for movie recommendations.
- Understand the application of Reinforcement Learning.
- Identify user preferences using movie genres.
- Generate personalized movie recommendations.
- Compare Q-Learning with a popularity-based recommendation system.
- Evaluate recommendation performance.

## Dataset

Dataset: MovieLens 1M

Source: GroupLens Research

Dataset Link:
https://grouplens.org/datasets/movielens/1m/

The dataset contains approximately:
- 1 million movie ratings.
- 6,000 users.
- 4,000 movies.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Reinforcement Learning
- Q-Learning

## Reinforcement Learning Components

- Agent: Movie recommendation system.
- Environment: MovieLens dataset and offline interaction simulation.
- State: User's dominant movie genre preference.
- Action: Movie selected for recommendation.
- Reward: Normalized historical movie rating.
- Next State: User preference state.
- Policy: Epsilon-greedy policy.

## Q-Learning Formula

Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]

Where:

- α = Learning rate
- γ = Discount factor
- r = Reward
- s = Current state
- a = Selected action
- s' = Next state

## Project Execution

### Step 1: Install Dependencies

Run:

python -m pip install -r requirements.txt

### Step 2: Run the Project

Run:

python movie_q_learning.py

### Step 3: Dataset

The MovieLens dataset is automatically downloaded and extracted into the data folder.

### Step 4: Results

The generated graphs, evaluation results and Q-table are saved in the outputs folder.

## Evaluation Metrics

- Hit Rate@10
- Precision@10
- Recall@10
- Coverage
- Average Rating of Hits

## Expected Outputs

- Rating distribution graph.
- Popular movies graph.
- Q-Learning training reward graph.
- Model comparison graph.
- Evaluation results CSV.
- Q-table.

## Conclusion

This project demonstrates the application of Q-Learning in personalized movie recommendations by learning from historical user ratings and estimating the value of recommending different movies.