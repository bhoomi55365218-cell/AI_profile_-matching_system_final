import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Profile Matching System",
    page_icon="🤝",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

USERS_FILE = BASE_DIR / "users.csv"
FEEDBACK_FILE = BASE_DIR / "feedback.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    users = pd.read_csv(USERS_FILE)
    feedback = pd.read_csv(FEEDBACK_FILE)

    return users, feedback


try:
    users, feedback = load_data()

except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}\n\n"
        "Make sure users.csv and feedback.csv are in the same "
        "folder as app.py."
    )
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_user_columns = [
    "user_id",
    "name",
    "age",
    "location",
    "profession",
    "experience_years",
    "professional_summary",
    "about_me",
    "mbti",
    "intrests"
]

required_feedback_columns = [
    "user_id",
    "matched_user_id",
    "action"
]


missing_user_columns = [
    column for column in required_user_columns
    if column not in users.columns
]

missing_feedback_columns = [
    column for column in required_feedback_columns
    if column not in feedback.columns
]


if missing_user_columns:
    st.error(
        "Missing columns in users.csv: "
        + ", ".join(missing_user_columns)
    )
    st.stop()


if missing_feedback_columns:
    st.error(
        "Missing columns in feedback.csv: "
        + ", ".join(missing_feedback_columns)
    )
    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

text_columns = [
    "professional_summary",
    "about_me",
    "intrests"
]

for column in text_columns:
    users[column] = users[column].fillna("").astype(str)

users["mbti"] = users["mbti"].fillna("").astype(str).str.upper()
users["location"] = users["location"].fillna("").astype(str)
users["profession"] = users["profession"].fillna("").astype(str)


# ============================================================
# CREATE PROFILE TEXT
# ============================================================

users["clean_text"] = (
    users["professional_summary"]
    + " "
    + users["about_me"]
    + " "
    + users["intrests"]
    + " "
    + users["profession"]
)


# ============================================================
# TF-IDF TEXT REPRESENTATION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    users["clean_text"]
)


# ============================================================
# TEXT SIMILARITY
# ============================================================

text_similarity = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# MBTI COMPATIBILITY
# ============================================================

mbti_groups = {
    "NT": ["INTJ", "INTP", "ENTJ", "ENTP"],
    "NF": ["INFJ", "INFP", "ENFJ", "ENFP"],
    "SJ": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
    "SP": ["ISTP", "ISFP", "ESTP", "ESFP"]
}


def get_mbti_group(mbti):

    mbti = str(mbti).upper()

    for group, types in mbti_groups.items():

        if mbti in types:
            return group

    return None


def mbti_compatibility(mbti1, mbti2):

    mbti1 = str(mbti1).upper()
    mbti2 = str(mbti2).upper()

    if not mbti1 or not mbti2:
        return 0.0

    if mbti1 == mbti2:
        return 1.0

    group1 = get_mbti_group(mbti1)
    group2 = get_mbti_group(mbti2)

    if group1 is not None and group1 == group2:
        return 0.7

    return 0.4


# ============================================================
# LOCATION COMPATIBILITY
# ============================================================

def location_compatibility(location1, location2):

    location1 = str(location1).strip().lower()
    location2 = str(location2).strip().lower()

    if not location1 or not location2:
        return 0.0

    if location1 == location2:
        return 1.0

    return 0.0


# ============================================================
# FIND USER INDEX
# ============================================================

def get_user_index(user_id):

    matches = users.index[
        users["user_id"].astype(str) == str(user_id)
    ]

    if len(matches) == 0:
        return None

    return matches[0]


# ============================================================
# CALCULATE PROFILE FEATURES
# ============================================================

def calculate_features(user_index, matched_index):

    text_score = float(
        text_similarity[user_index][matched_index]
    )

    mbti_score = mbti_compatibility(
        users.iloc[user_index]["mbti"],
        users.iloc[matched_index]["mbti"]
    )

    location_score = location_compatibility(
        users.iloc[user_index]["location"],
        users.iloc[matched_index]["location"]
    )

    return (
        text_score,
        mbti_score,
        location_score
    )


# ============================================================
# PREPARE FEEDBACK DATA
# ============================================================

feedback_data = []

for _, row in feedback.iterrows():

    user_index = get_user_index(row["user_id"])
    matched_index = get_user_index(row["matched_user_id"])

    if user_index is None or matched_index is None:
        continue

    try:
        action = int(row["action"])
    except (ValueError, TypeError):
        continue

    text_score, mbti_score, location_score = calculate_features(
        user_index,
        matched_index
    )

    feedback_data.append({
        "text_score": text_score,
        "mbti_score": mbti_score,
        "location_score": location_score,
        "action": action
    })


feedback_df = pd.DataFrame(feedback_data)


# ============================================================
# TRAIN AI FEEDBACK MODEL
# ============================================================

model = None
scaler = None
learned_weights = None
model_accuracy = None


if (
    not feedback_df.empty
    and len(feedback_df) >= 10
    and feedback_df["action"].nunique() >= 2
):

    X = feedback_df[
        [
            "text_score",
            "mbti_score",
            "location_score"
        ]
    ]

    y = feedback_df["action"]

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        model_accuracy = model.score(
            X_test_scaled,
            y_test
        )

        coefficients = abs(model.coef_[0])

        if coefficients.sum() > 0:

            learned_weights = (
                coefficients / coefficients.sum()
            )

    except ValueError:

        model = None


# ============================================================
# FALLBACK WEIGHTS
# ============================================================

if learned_weights is None:

    learned_weights = [0.50, 0.30, 0.20]


# ============================================================
# MATCH SCORE
# ============================================================

def calculate_match_score(
    user_index,
    matched_index
):

    text_score, mbti_score, location_score = (
        calculate_features(
            user_index,
            matched_index
        )
    )

    score = (
        learned_weights[0] * text_score
        + learned_weights[1] * mbti_score
        + learned_weights[2] * location_score
    )

    return {
        "total": float(score),
        "text": float(text_score),
        "mbti": float(mbti_score),
        "location": float(location_score)
    }


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def get_recommendations(
    selected_user_id,
    number_of_matches=5
):

    user_index = get_user_index(
        selected_user_id
    )

    if user_index is None:
        return pd.DataFrame()

    recommendations = []

    for matched_index in range(len(users)):

        if matched_index == user_index:
            continue

        result = calculate_match_score(
            user_index,
            matched_index
        )

        recommendations.append({
            "index": matched_index,
            "score": result["total"],
            "text_score": result["text"],
            "mbti_score": result["mbti"],
            "location_score": result["location"]
        })

    recommendations.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_matches = recommendations[
        :number_of_matches
    ]

    result_rows = []

    for match in top_matches:

        row = users.iloc[
            match["index"]
        ].copy()

        row["match_score"] = (
            match["score"] * 100
        )

        row["text_score"] = (
            match["text_score"] * 100
        )

        row["mbti_score"] = (
            match["mbti_score"] * 100
        )

        row["location_score"] = (
            match["location_score"] * 100
        )

        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("🤝 AI Profile Matching System")

st.markdown(
    """
    ### Find the most compatible profiles using AI

    The system combines:

    - 🧠 **TF-IDF + Cosine Similarity** for profile/content similarity
    - 🧩 **MBTI compatibility**
    - 📍 **Location compatibility**
    - 🤖 **Feedback-based Logistic Regression**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ System Information")

st.sidebar.write(
    f"**Total Profiles:** {len(users)}"
)

st.sidebar.write(
    f"**Feedback Records:** {len(feedback)}"
)

if model_accuracy is not None:

    st.sidebar.write(
        f"**AI Model Accuracy:** "
        f"{model_accuracy * 100:.2f}%"
    )

else:

    st.sidebar.write(
        "**AI Model:** Fallback weighting"
    )


st.sidebar.divider()

st.sidebar.subheader(
    "Learned Matching Weights"
)

st.sidebar.write(
    f"Text Similarity: "
    f"{learned_weights[0] * 100:.1f}%"
)

st.sidebar.write(
    f"MBTI Compatibility: "
    f"{learned_weights[1] * 100:.1f}%"
)

st.sidebar.write(
    f"Location Compatibility: "
    f"{learned_weights[2] * 100:.1f}%"
)


# ============================================================
# USER SELECTION
# ============================================================

st.subheader("👤 Select Your Profile")

user_options = users[
    [
        "user_id",
        "name",
        "profession",
        "location"
    ]
].copy()

user_options["display"] = (
    user_options["name"]
    + " — "
    + user_options["profession"]
    + " — "
    + user_options["location"]
)


selected_display = st.selectbox(
    "Choose a profile",
    user_options["display"].tolist()
)


selected_row = user_options[
    user_options["display"] == selected_display
].iloc[0]


selected_user_id = selected_row["user_id"]


# ============================================================
# PROFILE INFORMATION
# ============================================================

selected_user = users[
    users["user_id"] == selected_user_id
].iloc[0]


st.subheader("📋 Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Name",
        selected_user["name"]
    )

with col2:
    st.metric(
        "Profession",
        selected_user["profession"]
    )

with col3:
    st.metric(
        "Location",
        selected_user["location"]
    )

with col4:
    st.metric(
        "MBTI",
        selected_user["mbti"]
    )


st.write(
    "**Professional Summary:**"
)

st.write(
    selected_user["professional_summary"]
)

st.write(
    "**Interests:**"
)

st.write(
    selected_user["intrests"]
)


# ============================================================
# NUMBER OF MATCHES
# ============================================================

number_of_matches = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=min(10, len(users) - 1),
    value=min(5, len(users) - 1)
)


# ============================================================
# GENERATE MATCHES
# ============================================================

if st.button(
    "🤖 Find Best Matches",
    type="primary"
):

    recommendations = get_recommendations(
        selected_user_id,
        number_of_matches
    )

    if recommendations.empty:

        st.warning(
            "No matching profiles found."
        )

    else:

        st.subheader(
            "🏆 Recommended Profiles"
        )

        for rank, (_, match) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            score = match["match_score"]

            with st.container():

                st.markdown(
                    f"### #{rank} {match['name']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"💼 **Profession:** "
                        f"{match['profession']}"
                    )

                    st.write(
                        f"📍 **Location:** "
                        f"{match['location']}"
                    )

                with col2:

                    st.write(
                        f"🧠 **MBTI:** "
                        f"{match['mbti']}"
                    )

                    st.write(
                        f"⭐ **Match Score:** "
                        f"{score:.2f}%"
                    )

                with col3:

                    st.write(
                        f"📝 **Text Similarity:** "
                        f"{match['text_score']:.2f}%"
                    )

                    st.write(
                        f"🧩 **MBTI Compatibility:** "
                        f"{match['mbti_score']:.2f}%"
                    )

                    st.write(
                        f"📍 **Location Compatibility:** "
                        f"{match['location_score']:.2f}%"
                    )

                st.progress(
                    min(score / 100, 1.0)
                )

                st.write(
                    "**Why this profile matches:**"
                )

                reasons = []

                if match["text_score"] >= 50:
                    reasons.append(
                        "Strong profile/content similarity"
                    )

                if match["mbti_score"] >= 70:
                    reasons.append(
                        "Compatible MBTI personality group"
                    )

                if match["location_score"] == 100:
                    reasons.append(
                        "Same location"
                    )

                if not reasons:
                    reasons.append(
                        "Overall compatibility based on learned profile features"
                    )

                for reason in reasons:
                    st.write(
                        f"• {reason}"
                    )

                st.divider()


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander(
    "📊 View Dataset"
):

    st.dataframe(
        users.drop(
            columns=["clean_text"],
            errors="ignore"
        ),
        use_container_width=True
    )
