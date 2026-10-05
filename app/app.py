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
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

USERS_FILE = BASE_DIR / "users.csv"
FEEDBACK_FILE = BASE_DIR / "feedback.csv"


# ============================================================
# LOAD USERS
# ============================================================

if not USERS_FILE.exists():
    st.error(f"Required file not found: {USERS_FILE}")
    st.info("Make sure users.csv is in the same folder as app.py.")
    st.stop()

users = pd.read_csv(USERS_FILE)


# ============================================================
# CHECK REQUIRED USER COLUMNS
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

missing_columns = [
    col for col in required_user_columns
    if col not in users.columns
]

if missing_columns:
    st.error(
        "Missing columns in users.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# CLEAN USER DATA
# ============================================================

text_columns = [
    "name",
    "location",
    "profession",
    "professional_summary",
    "about_me",
    "mbti",
    "intrests"
]

for col in text_columns:
    users[col] = users[col].fillna("").astype(str).str.strip()

users["user_id"] = users["user_id"].astype(str)

users["mbti"] = users["mbti"].str.upper()

users["location"] = users["location"].str.lower()

users["profession"] = users["profession"].str.lower()


# ============================================================
# CREATE PROFILE TEXT
# ============================================================

users["profile_text"] = (
    users["professional_summary"] + " "
    + users["about_me"] + " "
    + users["intrests"] + " "
    + users["profession"]
)


# ============================================================
# TF-IDF PROFILE SIMILARITY
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    users["profile_text"]
)

text_similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# MBTI COMPATIBILITY
# ============================================================

def mbti_group(mbti):
    """
    Groups MBTI types into four broad categories.
    """

    mbti = str(mbti).upper()

    if mbti.endswith("NT"):
        return "NT"

    if mbti.endswith("NF"):
        return "NF"

    if mbti.endswith("SJ"):
        return "SJ"

    if mbti.endswith("SP"):
        return "SP"

    return "UNKNOWN"


def mbti_compatibility(mbti1, mbti2):

    mbti1 = str(mbti1).upper()
    mbti2 = str(mbti2).upper()

    if not mbti1 or not mbti2:
        return 0.0

    if mbti1 == mbti2:
        return 1.0

    group1 = mbti_group(mbti1)
    group2 = mbti_group(mbti2)

    if group1 == "UNKNOWN" or group2 == "UNKNOWN":
        return 0.0

    if group1 == group2:
        return 0.7

    return 0.4


# ============================================================
# LOCATION COMPATIBILITY
# ============================================================

def location_compatibility(location1, location2):

    location1 = str(location1).lower().strip()
    location2 = str(location2).lower().strip()

    if location1 and location1 == location2:
        return 1.0

    return 0.0


# ============================================================
# LOAD / CREATE FEEDBACK FILE
# ============================================================

if FEEDBACK_FILE.exists():

    feedback = pd.read_csv(FEEDBACK_FILE)

else:

    feedback = pd.DataFrame(
        columns=[
            "user_id",
            "matched_user_id",
            "action"
        ]
    )

    feedback.to_csv(
        FEEDBACK_FILE,
        index=False
    )


# ============================================================
# CHECK FEEDBACK COLUMNS
# ============================================================

required_feedback_columns = [
    "user_id",
    "matched_user_id",
    "action"
]

for col in required_feedback_columns:

    if col not in feedback.columns:
        feedback[col] = ""


feedback["user_id"] = feedback["user_id"].astype(str)
feedback["matched_user_id"] = feedback["matched_user_id"].astype(str)
feedback["action"] = feedback["action"].astype(str).str.lower()


# ============================================================
# SAVE FEEDBACK FUNCTION
# ============================================================

def save_feedback(user_id, matched_user_id, action):

    new_feedback = pd.DataFrame(
        [{
            "user_id": str(user_id),
            "matched_user_id": str(matched_user_id),
            "action": action
        }]
    )

    new_feedback.to_csv(
        FEEDBACK_FILE,
        mode="a",
        header=not FEEDBACK_FILE.exists()
        or FEEDBACK_FILE.stat().st_size == 0,
        index=False
    )


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

def train_feedback_model():

    if feedback.empty:
        return None, [0.50, 0.30, 0.20], None

    feature_rows = []
    labels = []

    for _, row in feedback.iterrows():

        user_id = str(row["user_id"])
        matched_id = str(row["matched_user_id"])
        action = str(row["action"]).lower()

        if user_id not in users["user_id"].values:
            continue

        if matched_id not in users["user_id"].values:
            continue

        if action not in ["accept", "reject"]:
            continue

        user_index = users.index[
            users["user_id"] == user_id
        ][0]

        matched_index = users.index[
            users["user_id"] == matched_id
        ][0]

        text_score = text_similarity_matrix[
            user_index,
            matched_index
        ]

        mbti_score = mbti_compatibility(
            users.loc[user_index, "mbti"],
            users.loc[matched_index, "mbti"]
        )

        location_score = location_compatibility(
            users.loc[user_index, "location"],
            users.loc[matched_index, "location"]
        )

        feature_rows.append([
            text_score,
            mbti_score,
            location_score
        ])

        labels.append(
            1 if action == "accept" else 0
        )

    if len(feature_rows) < 10:
        return None, [0.50, 0.30, 0.20], None

    if len(set(labels)) < 2:
        return None, [0.50, 0.30, 0.20], None

    X = pd.DataFrame(
        feature_rows,
        columns=[
            "text_similarity",
            "mbti_compatibility",
            "location_compatibility"
        ]
    )

    y = pd.Series(labels)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = LogisticRegression(
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        accuracy = model.score(
            X_test,
            y_test
        )

        coefficients = abs(
            model.coef_[0]
        )

        if coefficients.sum() == 0:

            weights = [
                0.50,
                0.30,
                0.20
            ]

        else:

            weights = (
                coefficients /
                coefficients.sum()
            ).tolist()

        return model, weights, accuracy

    except Exception:
        return None, [0.50, 0.30, 0.20], None


model, weights, model_accuracy = train_feedback_model()


# ============================================================
# MATCHING FUNCTION
# ============================================================

def calculate_matches(selected_user_id, number_of_matches):

    selected_index = users.index[
        users["user_id"] == selected_user_id
    ][0]

    results = []

    for index, candidate in users.iterrows():

        candidate_id = str(candidate["user_id"])

        # Don't recommend the same person
        if candidate_id == selected_user_id:
            continue

        text_score = text_similarity_matrix[
            selected_index,
            index
        ]

        mbti_score = mbti_compatibility(
            users.loc[selected_index, "mbti"],
            candidate["mbti"]
        )

        location_score = location_compatibility(
            users.loc[selected_index, "location"],
            candidate["location"]
        )

        final_score = (
            weights[0] * text_score
            + weights[1] * mbti_score
            + weights[2] * location_score
        )

        results.append({
            "index": index,
            "user_id": candidate_id,
            "name": candidate["name"],
            "score": final_score,
            "text_score": text_score,
            "mbti_score": mbti_score,
            "location_score": location_score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:number_of_matches]


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Profile Matching System")

st.write(
    "Find the most compatible profiles using "
    "AI-based profile similarity, personality compatibility, "
    "and location matching."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Matching Model")

    st.write(
        f"**Profile similarity:** "
        f"{weights[0] * 100:.1f}%"
    )

    st.write(
        f"**MBTI compatibility:** "
        f"{weights[1] * 100:.1f}%"
    )

    st.write(
        f"**Location compatibility:** "
        f"{weights[2] * 100:.1f}%"
    )

    if model_accuracy is not None:

        st.write(
            f"**Feedback model accuracy:** "
            f"{model_accuracy * 100:.1f}%"
        )

    else:

        st.info(
            "The system is currently using "
            "default matching weights. "
            "More feedback is required to train "
            "the feedback model."
        )


# ============================================================
# USER SELECTION
# ============================================================

user_options = users["user_id"].tolist()

selected_user_id = st.selectbox(
    "Select your profile",
    user_options,
    format_func=lambda uid: (
        f"{users.loc[users['user_id'] == uid, 'name'].iloc[0]}"
        f" — ID {uid}"
    )
)


selected_user = users[
    users["user_id"] == selected_user_id
].iloc[0]


# ============================================================
# SELECTED USER PROFILE
# ============================================================

st.subheader("👤 Your Profile")

col1, col2, col3 = st.columns(3)

with col1:

    st.write(
        f"**Name:** {selected_user['name']}"
    )

    st.write(
        f"**Age:** {selected_user['age']}"
    )

    st.write(
        f"**Location:** {selected_user['location']}"
    )

with col2:

    st.write(
        f"**Profession:** {selected_user['profession']}"
    )

    st.write(
        f"**Experience:** "
        f"{selected_user['experience_years']} years"
    )

    st.write(
        f"**MBTI:** {selected_user['mbti']}"
    )

with col3:

    st.write(
        f"**Interests:** {selected_user['intrests']}"
    )


st.write(
    f"**About:** {selected_user['about_me']}"
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
# FIND MATCHES
# ============================================================

if st.button(
    "🔎 Find Best Matches",
    type="primary"
):

    st.session_state["matches"] = calculate_matches(
        selected_user_id,
        number_of_matches
    )


# ============================================================
# DISPLAY MATCHES
# ============================================================

if "matches" in st.session_state:

    st.subheader("🎯 Recommended Profiles")

    matches = st.session_state["matches"]

    if not matches:

        st.warning(
            "No matching profiles found."
        )

    else:

        for rank, match in enumerate(
            matches,
            start=1
        ):

            candidate = users[
                users["user_id"] == match["user_id"]
            ].iloc[0]

            st.markdown("---")

            st.subheader(
                f"#{rank} — {candidate['name']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"📍 **Location:** "
                    f"{candidate['location']}"
                )

                st.write(
                    f"💼 **Profession:** "
                    f"{candidate['profession']}"
                )

            with col2:

                st.write(
                    f"🧠 **MBTI:** "
                    f"{candidate['mbti']}"
                )

                st.write(
                    f"⭐ **Experience:** "
                    f"{candidate['experience_years']} years"
                )

            with col3:

                st.metric(
                    "Match Score",
                    f"{match['score'] * 100:.1f}%"
                )

            st.write(
                f"**Professional Summary:** "
                f"{candidate['professional_summary']}"
            )

            st.write(
                f"**About:** "
                f"{candidate['about_me']}"
            )

            st.write(
                f"**Interests:** "
                f"{candidate['intrests']}"
            )

            # ------------------------------------------------
            # MATCH COMPONENTS
            # ------------------------------------------------

            st.write("### 📊 Match Analysis")

            score_col1, score_col2, score_col3 = st.columns(3)

            with score_col1:

                st.metric(
                    "Profile Similarity",
                    f"{match['text_score'] * 100:.1f}%"
                )

            with score_col2:

                st.metric(
                    "MBTI Compatibility",
                    f"{match['mbti_score'] * 100:.1f}%"
                )

            with score_col3:

                st.metric(
                    "Location Match",
                    f"{match['location_score'] * 100:.1f}%"
                )

            # ------------------------------------------------
            # REASONS
            # ------------------------------------------------

            reasons = []

            if match["text_score"] >= 0.50:

                reasons.append(
                    "Your interests and profile are highly similar."
                )

            elif match["text_score"] >= 0.25:

                reasons.append(
                    "Your profiles have some common interests."
                )

            if match["mbti_score"] >= 1.0:

                reasons.append(
                    "You have the same MBTI personality type."
                )

            elif match["mbti_score"] >= 0.7:

                reasons.append(
                    "Your MBTI personality groups are similar."
                )

            if match["location_score"] == 1.0:

                reasons.append(
                    "You are from the same location."
                )

            if reasons:

                st.write("### 💡 Why this is a match")

                for reason in reasons:

                    st.write(
                        f"• {reason}"
                    )

            # ------------------------------------------------
            # ACCEPT / REJECT BUTTONS
            # ------------------------------------------------

            st.write("### 🤝 Your Decision")

            accept_col, reject_col = st.columns(2)

            with accept_col:

                if st.button(
                    "✅ Accept",
                    key=f"accept_{selected_user_id}_{match['user_id']}"
                ):

                    save_feedback(
                        selected_user_id,
                        match["user_id"],
                        "accept"
                    )

                    st.success(
                        f"You accepted {candidate['name']}!"
                    )

                    st.toast(
                        "Match accepted! 👍"
                    )

            with reject_col:

                if st.button(
                    "❌ Reject",
                    key=f"reject_{selected_user_id}_{match['user_id']}"
                ):

                    save_feedback(
                        selected_user_id,
                        match["user_id"],
                        "reject"
                    )

                    st.warning(
                        f"You rejected {candidate['name']}."
                    )

                    st.toast(
                        "Match rejected."
                    )


# ============================================================
# FEEDBACK DATA
# ============================================================

st.markdown("---")

with st.expander("📈 View Feedback Data"):

    if FEEDBACK_FILE.exists():

        current_feedback = pd.read_csv(
            FEEDBACK_FILE
        )

        if current_feedback.empty:

            st.info(
                "No feedback has been recorded yet."
            )

        else:

            st.dataframe(
                current_feedback,
                use_container_width=True
            )

    else:

        st.info(
            "No feedback file found."
        )


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander("📋 View Users Dataset"):

    st.dataframe(
        users.drop(
            columns=["profile_text"],
            errors="ignore"
        ),
        use_container_width=True
    )
