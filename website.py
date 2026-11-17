# ============================================================
# PERSONALIZED MOVIE RECOMMENDATION WEBSITE
# Save this file as: website.py
# Keep it in the same folder as movie_q_learning.py.
# Expected structure:
# project/
#   website.py
#   movie_q_learning.py
#   outputs/
# ============================================================

from pathlib import Path
import base64
import numpy as np
import pandas as pd
import streamlit as st


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load background image. The actual background CSS is applied AFTER the main
# stylesheet below so that the main .stApp rule cannot overwrite it.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
BG_PATH = BASE_DIR / "bg.jpg"

encoded_image = None
if BG_PATH.exists():
    encoded_image = base64.b64encode(BG_PATH.read_bytes()).decode("utf-8")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(116, 91, 160, .18), transparent 28%),
            linear-gradient(135deg, #10101b 0%, #171625 52%, #11131d 100%);
        color: #f5f1fa;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #171624;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    h1, h2, h3 {
        color: #f7f1ff !important;
    }
    .hero {
        padding: 34px 34px 30px 34px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 24px;
        background: linear-gradient(115deg, rgba(105,77,145,.34), rgba(36,35,57,.82));
        margin-bottom: 24px;
    }
    .eyebrow {
        color: #d4b8f5;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(32px, 5vw, 54px);
        line-height: 1.08;
        margin: 8px 0 12px 0;
        color: #fff;
    }
    .hero-copy {
        color: #d2cbdc;
        max-width: 680px;
        font-size: 15px;
        line-height: 1.7;
    }
    .section-label {
        color: #d9c4f5;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 8px 0 12px 0;
    }
    .metric-card {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 17px;
        padding: 17px 18px;
        min-height: 104px;
    }
    .metric-name { color: #bdb4ca; font-size: 12px; }
    .metric-value { color: #fff; font-size: 27px; font-weight: 700; margin-top: 7px; }
    .movie-card {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 15px;
        padding: 15px;
        min-height: 118px;
        margin-bottom: 10px;
    }
    .movie-title { font-weight: 700; color: #fff; font-size: 15px; }
    .movie-meta { color: #c4bbd1; font-size: 12px; margin-top: 7px; }
    .tag {
        display: inline-block;
        border-radius: 30px;
        background: rgba(190,151,235,.16);
        color: #e4d1fb;
        padding: 4px 9px;
        font-size: 11px;
        margin: 7px 4px 0 0;
    }
    .info-box {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 15px;
        padding: 16px 18px;
        color: #d2cada;
        line-height: 1.65;
    }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #a98bd0;
        background: #8d6bb8;
        color: white;
        font-weight: 700;
        min-height: 42px;
    }
    div.stButton > button:hover {
        background: #a181cc;
        border-color: #cbb1ec;
        color: white;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 15px;
        padding: 12px 15px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
    }
    hr { border-color: rgba(255,255,255,.12); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply the background AFTER the main stylesheet to prevent it being overridden.
if encoded_image:
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(rgba(0, 0, 0, 0.42), rgba(0, 0, 0, 0.52)),
                url("data:image/jpeg;base64,{encoded_image}") center center / cover fixed !important;
            background-repeat: no-repeat !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0.08) !important;
        }}
        [data-testid="stMain"] {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Background image not found. Place bg.jpg here: {BG_PATH}")


# -------------------- DATA LOADING --------------------

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data
def load_npy(path_string):
    path = Path(path_string)
    if path.exists():
        try:
            return np.load(path, allow_pickle=False)
        except Exception:
            return None
    return None


def find_image(filename):
    """Find a generated graph in outputs/."""
    path = OUTPUT_DIR / filename
    return path if path.exists() else None


evaluation = load_csv(str(OUTPUT_DIR / "evaluation_results.csv"))
movies = load_csv(str(OUTPUT_DIR / "movie_mapping.csv"))
q_table = load_npy(str(OUTPUT_DIR / "q_table.npy"))

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("A Q-Learning Movie Recommendation Agent")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Discover Movies", "Model Performance", "Project Insights", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Project**")
    st.caption("Personalized Movie Recommendation Agent Using Q-Learning")
    st.caption("Advanced Machine Learning")


# -------------------- HERO --------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">A smarter way to choose your next movie</div>
        <div class="hero-title">Your next favourite<br>movie starts here.</div>
        <div class="hero-copy">
            CineMatch uses a Q-Learning recommendation model trained on MovieLens
            ratings to explore movie preferences and present recommendations.
            Browse the project results, explore genres and view model performance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- DATA STATUS --------------------

missing_files = []
for filename in [
    "evaluation_results.csv",
    "movie_mapping.csv",
    "q_table.npy",
]:
    if not (OUTPUT_DIR / filename).exists():
        missing_files.append(filename)

if missing_files:
    st.warning(
        "Some output files are missing from the outputs folder: "
        + ", ".join(missing_files)
        + ". Run movie_q_learning.py first, then refresh this website."
    )


# -------------------- HELPER FUNCTIONS --------------------

def safe_text(value, fallback="Not available"):
    if pd.isna(value):
        return fallback
    return str(value)


def get_genre_list(movie_frame):
    all_genres = set()
    if movie_frame is not None and "Genres" in movie_frame.columns:
        for value in movie_frame["Genres"].dropna():
            all_genres.update(str(value).split("|"))
    return sorted(g for g in all_genres if g)


def get_recommendations(selected_genre, count):
    """
    The saved Q-table rows correspond to the sorted genre list used
    by the training script, followed by an unknown-state row.
    """
    if movies is None or q_table is None:
        return pd.DataFrame()

    required = {"MovieID", "Title", "Genres"}
    if not required.issubset(set(movies.columns)):
        return pd.DataFrame()

    genre_list = get_genre_list(movies)
    if selected_genre == "All genres":
        # Use the best Q-value across genre states for a broad discovery list.
        state_rows = min(len(genre_list), q_table.shape[0])
        if state_rows <= 0:
            scores = q_table[-1]
        else:
            scores = np.max(q_table[:state_rows], axis=0)
    else:
        if selected_genre not in genre_list:
            return pd.DataFrame()
        state_index = genre_list.index(selected_genre)
        if state_index >= q_table.shape[0]:
            return pd.DataFrame()
        scores = q_table[state_index]

    if len(scores) != len(movies):
        return pd.DataFrame()

    result = movies.copy()
    result["Q-Value"] = scores[:len(result)]
    result = result.sort_values("Q-Value", ascending=False)
    if selected_genre != "All genres":
        result = result[
            result["Genres"].fillna("").str.contains(
                selected_genre, regex=False
            )
        ]
    return result.head(count).reset_index(drop=True)


def movie_card(row, rank=None):
    title = safe_text(row.get("Title", "Unknown movie"))
    genres = safe_text(row.get("Genres", "Genre unavailable"))
    q_value = row.get("Q-Value", None)
    q_text = f"Q-value: {q_value:.3f}" if pd.notna(q_value) else ""
    rank_text = f"#{rank} · " if rank is not None else ""
    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{rank_text}{title}</div>
            <div class="movie-meta">{genres}</div>
            <span class="tag">Q-Learning pick</span>
            <span class="tag">{q_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- PAGE: DISCOVER MOVIES --------------------

if page == "Discover Movies":
    st.markdown('<div class="section-label">Personalized discovery</div>', unsafe_allow_html=True)

    if movies is None or q_table is None:
        st.error(
            "The movie mapping or Q-table could not be loaded. "
            "Make sure movie_mapping.csv and q_table.npy are inside outputs/."
        )
    else:
        genre_options = ["All genres"] + get_genre_list(movies)
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_genre = st.selectbox(
                "Choose a genre to explore",
                genre_options,
            )
        with col2:
            number_to_show = st.select_slider(
                "Number of recommendations",
                options=[5, 10, 15, 20],
                value=10,
            )

        recommendations = get_recommendations(selected_genre, number_to_show)

        if recommendations.empty:
            st.info("No recommendations could be generated for this selection.")
        else:
            st.markdown(
                f"### Recommended for {selected_genre}"
            )
            st.caption(
                "Ranked using the saved Q-table values. These are model-generated "
                "recommendations, not live streaming availability."
            )
            left, right = st.columns(2)
            for index, (_, row) in enumerate(recommendations.iterrows()):
                with (left if index % 2 == 0 else right):
                    movie_card(row, index + 1)

        st.markdown("---")
        st.markdown("### Search the movie collection")
        search_term = st.text_input(
            "Search by movie title",
            placeholder="Try: Toy Story, Matrix, Titanic...",
        )
        if search_term.strip():
            matches = movies[
                movies["Title"].fillna("").str.contains(
                    search_term.strip(), case=False, regex=False
                )
            ].head(30)
            if matches.empty:
                st.info("No matching movie titles found.")
            else:
                st.dataframe(
                    matches[["Title", "Genres"]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.caption(f"{len(movies):,} movies available in the movie mapping.")


# -------------------- PAGE: MODEL PERFORMANCE --------------------

elif page == "Model Performance":
    st.markdown('<div class="section-label">Evaluation dashboard</div>', unsafe_allow_html=True)

    if evaluation is None or evaluation.empty:
        st.info("evaluation_results.csv was not found or contains no results.")
    else:
        st.markdown("### Model comparison")
        st.dataframe(evaluation, use_container_width=True, hide_index=True)

        numeric_columns = evaluation.select_dtypes(include="number").columns.tolist()
        if numeric_columns and "Model" in evaluation.columns:
            metric = st.selectbox("Choose a metric to compare", numeric_columns)
            chart_data = evaluation[["Model", metric]].dropna().set_index("Model")
            st.bar_chart(chart_data, use_container_width=True)
            st.caption(
                "Use the CSV table for exact values. Higher values are generally "
                "better for the listed Hit Rate, Precision, Recall and Coverage metrics."
            )

    st.markdown("---")
    st.markdown("### Generated evaluation visualisation")
    comparison_image = find_image("model_comparison.png")
    if comparison_image:
        st.image(
            str(comparison_image),
            caption="Model comparison generated by the Python project",
            use_container_width=True,
        )
    else:
        st.info("model_comparison.png is not present in outputs/ yet.")


# -------------------- PAGE: PROJECT INSIGHTS --------------------

elif page == "Project Insights":
    st.markdown('<div class="section-label">Dataset and training insights</div>', unsafe_allow_html=True)

    if movies is not None:
        st.metric("Movies in mapping file", f"{len(movies):,}")
    if evaluation is not None:
        st.metric("Models evaluated", f"{len(evaluation):,}")

    graph_files = [
        ("rating_distribution.png", "Movie rating distribution"),
        ("popular_movies.png", "Top 10 most-rated movies"),
        ("training_rewards.png", "Q-Learning training rewards"),
        ("model_comparison.png", "Recommendation model comparison"),
    ]

    for start in range(0, len(graph_files), 2):
        cols = st.columns(2)
        for col, (filename, caption) in zip(cols, graph_files[start:start + 2]):
            with col:
                image_path = find_image(filename)
                st.markdown(f"### {caption}")
                if image_path:
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.info(f"{filename} has not been generated yet.")

    st.markdown("---")
    st.markdown("### Saved Q-table")
    if q_table is not None:
        a, b = st.columns(2)
        a.metric("Q-table states", q_table.shape[0])
        b.metric("Q-table actions", q_table.shape[1])
        st.caption(
            "Rows represent preference states and columns represent movie actions "
            "according to the Python implementation."
        )
    else:
        st.info("q_table.npy was not found in outputs/.")


# -------------------- PAGE: ABOUT --------------------

elif page == "About":
    st.markdown('<div class="section-label">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### Personalized Movie Recommendation Agent Using Q-Learning

        This mini project demonstrates a reinforcement-learning approach to
        movie recommendation using the MovieLens 1M dataset.

        **Core concepts**
        - **State:** A user's dominant genre preference.
        - **Action:** Selecting a movie to recommend.
        - **Reward:** A transformed rating, calculated as `(rating - 3) / 2`.
        - **Policy:** An epsilon-greedy strategy used in action selection.
        - **Learning:** Q-values are updated using the Q-learning update equation.

        **Technology used**
        - Python
        - Pandas and NumPy
        - Matplotlib
        - Scikit-learn
        - Streamlit

        **Important:** This website displays the saved outputs of the Python
        project. It does not retrain the model when the page is opened.
        """
    )

    st.markdown(
        """
        <div class="info-box">
            <b>How to run</b><br>
            1. Keep <code>website.py</code> in the main project folder.<br>
            2. Keep all generated CSV, NPY and PNG files inside <code>outputs/</code>.<br>
            3. Open a terminal in the project folder.<br>
            4. Run <code>python -m pip install streamlit</code>.<br>
            5. Run <code>python -m streamlit run website.py</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- FOOTER --------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#a99fb5; font-size:12px; padding:10px 0 18px;">
        CINEMATCH · Personalized Movie Recommendation Agent · Powered by Q-Learning
    </div>
    """,
    unsafe_allow_html=True,
)
