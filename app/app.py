import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


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

    st.error(
        f"Required file not found: {USERS_FILE}"
    )

    st.info(
        "Make sure users.csv is in the same folder as app.py."
    )

    st.stop()


users = pd.read_csv(USERS_FILE)


# ============================================================
# REQUIRED USER COLUMNS
# ============================================================

REQUIRED_USER_COLUMNS = [
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
    column
    for column in REQUIRED_USER_COLUMNS
    if column not in users.columns
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

TEXT_COLUMNS = [
    "name",
    "location",
    "profession",
    "professional_summary",
    "about_me",
    "mbti",
    "intrests"
]


for column in TEXT_COLUMNS:

    users[column] = (
        users[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


users["user_id"] = (
    users["user_id"]
    .astype(str)
    .str.strip()
)


users["mbti"] = (
    users["mbti"]
    .str.upper()
    .str.strip()
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CREATE COMBINED PROFILE TEXT
# ============================================================

users["combined_text"] = (
    users["professional_summary"].fillna("")
    + " "
    + users["about_me"].fillna("")
    + " "
    + users["intrests"].fillna("")
    + " "
    + users["profession"].fillna("")
)


users["clean_text"] = (
    users["combined_text"]
    .apply(clean_text)
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)


tfidf_matrix = vectorizer.fit_transform(
    users["clean_text"]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

text_similarity = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# MBTI GROUPS
# ============================================================

MBTI_GROUPS = {

    "NT": [
        "INTJ",
        "INTP",
        "ENTJ",
        "ENTP"
    ],

    "NF": [
        "INFJ",
        "INFP",
        "ENFJ",
        "ENFP"
    ],

    "SJ": [
        "ISTJ",
        "ISFJ",
        "ESTJ",
        "ESFJ"
    ],

    "SP": [
        "ISTP",
        "ISFP",
        "ESTP",
        "ESFP"
    ]
}


def get_mbti_group(mbti):

    mbti = str(mbti).upper().strip()

    for group, types in MBTI_GROUPS.items():

        if mbti in types:
            return group

    return None


# ============================================================
# MBTI COMPATIBILITY
# ============================================================

def mbti_compatibility(mbti1, mbti2):

    mbti1 = str(mbti1).upper().strip()
    mbti2 = str(mbti2).upper().strip()

    # Same personality
    if mbti1 == mbti2:
        return 1.0

    group1 = get_mbti_group(mbti1)
    group2 = get_mbti_group(mbti2)

    # Same personality group
    if (
        group1 is not None
        and group1 == group2
    ):
        return 0.7

    # Different group
    return 0.4


# ============================================================
# LOCATION COMPATIBILITY
# ============================================================

def location_compatibility(
    location1,
    location2
):

    location1 = (
        str(location1)
        .strip()
        .lower()
    )

    location2 = (
        str(location2)
        .strip()
        .lower()
    )

    if location1 == location2:
        return 1.0

    return 0.5


# ============================================================
# FEEDBACK FILE
# ============================================================

if FEEDBACK_FILE.exists():

    feedback = pd.read_csv(
        FEEDBACK_FILE
    )

else:

    feedback = pd.DataFrame(
        columns=[
            "user_id",
            "matched_user_id",
            "action",
            "timestamp"
        ]
    )

    feedback.to_csv(
        FEEDBACK_FILE,
        index=False
    )


# ============================================================
# NORMALIZE FEEDBACK COLUMNS
# ============================================================

if "user_id" not in feedback.columns:
    feedback["user_id"] = ""

if "matched_user_id" not in feedback.columns:
    feedback["matched_user_id"] = ""

if "action" not in feedback.columns:
    feedback["action"] = ""

if "timestamp" not in feedback.columns:
    feedback["timestamp"] = ""


feedback["user_id"] = (
    feedback["user_id"]
    .astype(str)
    .str.strip()
)


feedback["matched_user_id"] = (
    feedback["matched_user_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT FEEDBACK ACTION
# ============================================================

def convert_action(action):

    value = str(action).strip().lower()

    if value in [
        "1",
        "accept",
        "accepted",
        "true"
    ]:
        return 1

    if value in [
        "0",
        "reject",
        "rejected",
        "false"
    ]:
        return 0

    return None


feedback["accepted"] = (
    feedback["action"]
    .apply(convert_action)
)


# ============================================================
# BASE WEIGHTS
# ============================================================

BASE_WEIGHTS = np.array(
    [
        0.50,   # Text similarity
        0.30,   # MBTI
        0.20    # Location
    ],
    dtype=float
)


FEATURES = [
    "text_similarity",
    "mbti_score",
    "location_score"
]


# ============================================================
# NORMALIZE WEIGHTS
# ============================================================

def normalize_weights(
    weights,
    fallback=None
):

    weights = np.abs(
        np.asarray(
            weights,
            dtype=float
        )
    )

    total = weights.sum()

    if total == 0:

        if fallback is not None:

            return np.asarray(
                fallback,
                dtype=float
            )

        return BASE_WEIGHTS.copy()

    return weights / total


# ============================================================
# CREATE FEATURES FROM FEEDBACK
# ============================================================

def create_feedback_features(
    feedback_df
):

    rows = []

    for _, row in feedback_df.iterrows():

        try:

            user_matches = users.index[
                users["user_id"]
                == str(row["user_id"])
            ]

            matched_matches = users.index[
                users["user_id"]
                == str(row["matched_user_id"])
            ]

            if (
                len(user_matches) == 0
                or
                len(matched_matches) == 0
            ):
                continue


            user_index = user_matches[0]

            matched_index = matched_matches[0]


            # ------------------------------------------------
            # Text similarity
            # ------------------------------------------------

            text_score = float(
                text_similarity[
                    user_index,
                    matched_index
                ]
            )


            # ------------------------------------------------
            # MBTI
            # ------------------------------------------------

            mbti_score = float(
                mbti_compatibility(
                    users.iloc[
                        user_index
                    ]["mbti"],

                    users.iloc[
                        matched_index
                    ]["mbti"]
                )
            )


            # ------------------------------------------------
            # Location
            # ------------------------------------------------

            location_score = float(
                location_compatibility(
                    users.iloc[
                        user_index
                    ]["location"],

                    users.iloc[
                        matched_index
                    ]["location"]
                )
            )


            accepted = convert_action(
                row["action"]
            )


            if accepted is None:
                continue


            rows.append({

                "user_id":
                    str(row["user_id"]),

                "matched_user_id":
                    str(row["matched_user_id"]),

                "text_similarity":
                    text_score,

                "mbti_score":
                    mbti_score,

                "location_score":
                    location_score,

                "accepted":
                    accepted
            })


        except Exception:

            continue


    return pd.DataFrame(rows)


# ============================================================
# LEARN DYNAMIC WEIGHTS
# ============================================================

def learn_dynamic_weights(
    feedback_df
):

    feature_data = create_feedback_features(
        feedback_df
    )


    # --------------------------------------------------------
    # Default global weights
    # --------------------------------------------------------

    global_weights = (
        BASE_WEIGHTS.copy()
    )


    # --------------------------------------------------------
    # User-specific weights
    # --------------------------------------------------------

    personalized_weights = {}


    # ========================================================
    # GLOBAL MODEL
    # ========================================================

    if (
        len(feature_data) >= 4
        and
        feature_data["accepted"].nunique() >= 2
    ):

        try:

            model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )


            model.fit(
                feature_data[FEATURES],
                feature_data["accepted"]
            )


            global_weights = normalize_weights(
                model.coef_[0],
                BASE_WEIGHTS
            )


        except Exception:

            global_weights = (
                BASE_WEIGHTS.copy()
            )


    # ========================================================
    # PERSONALIZED USER MODEL
    # ========================================================

    if not feature_data.empty:

        for user_id, group in (
            feature_data.groupby("user_id")
        ):

            # Need enough examples
            if len(group) < 4:
                continue


            # Need both Accept and Reject
            if group["accepted"].nunique() < 2:
                continue


            try:

                user_model = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )


                user_model.fit(
                    group[FEATURES],
                    group["accepted"]
                )


                personalized_weights[
                    str(user_id)
                ] = normalize_weights(
                    user_model.coef_[0],
                    global_weights
                )


            except Exception:

                continue


    return (
        global_weights,
        personalized_weights,
        feature_data
    )


# ============================================================
# TRAIN ADAPTIVE SYSTEM
# ============================================================

(
    dynamic_global_weights,
    dynamic_user_weights,
    dynamic_feedback_features
) = learn_dynamic_weights(
    feedback
)


# ============================================================
# CALCULATE ADAPTIVE MATCH SCORE
# ============================================================

def calculate_match_components(
    user_id,
    matched_user_id
):

    user_matches = users.index[
        users["user_id"]
        == str(user_id)
    ]

    matched_matches = users.index[
        users["user_id"]
        == str(matched_user_id)
    ]


    if (
        len(user_matches) == 0
        or
        len(matched_matches) == 0
    ):

        return None


    user_index = user_matches[0]

    matched_index = matched_matches[0]


    # --------------------------------------------------------
    # Text score
    # --------------------------------------------------------

    text_score = float(
        text_similarity[
            user_index,
            matched_index
        ]
    )


    # --------------------------------------------------------
    # MBTI score
    # --------------------------------------------------------

    mbti_score = float(
        mbti_compatibility(
            users.iloc[
                user_index
            ]["mbti"],

            users.iloc[
                matched_index
            ]["mbti"]
        )
    )


    # --------------------------------------------------------
    # Location score
    # --------------------------------------------------------

    location_score = float(
        location_compatibility(
            users.iloc[
                user_index
            ]["location"],

            users.iloc[
                matched_index
            ]["location"]
        )
    )


    # --------------------------------------------------------
    # Get learned weights
    # --------------------------------------------------------

    weights = dynamic_user_weights.get(
        str(user_id),
        dynamic_global_weights
    )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    total_score = (

        weights[0] * text_score

        +

        weights[1] * mbti_score

        +

        weights[2] * location_score

    )


    return {

        "score":
            round(
                total_score * 100,
                2
            ),

        "text_score":
            round(
                text_score * 100,
                2
            ),

        "mbti_score":
            round(
                mbti_score * 100,
                2
            ),

        "location_score":
            round(
                location_score * 100,
                2
            ),

        "weights":
            weights
    }


# ============================================================
# GET TOP MATCHES
# ============================================================

def get_top_matches(
    selected_user_id,
    number_of_matches=5
):

    results = []


    for _, candidate in users.iterrows():

        candidate_id = str(
            candidate["user_id"]
        )


        # Don't recommend yourself
        if candidate_id == str(
            selected_user_id
        ):
            continue


        components = calculate_match_components(
            selected_user_id,
            candidate_id
        )


        if components is None:
            continue


        results.append({

            "user_id":
                candidate_id,

            "name":
                candidate["name"],

            "age":
                candidate["age"],

            "location":
                candidate["location"],

            "profession":
                candidate["profession"],

            "experience_years":
                candidate["experience_years"],

            "professional_summary":
                candidate[
                    "professional_summary"
                ],

            "about_me":
                candidate["about_me"],

            "mbti":
                candidate["mbti"],

            "intrests":
                candidate["intrests"],

            "score":
                components["score"],

            "text_score":
                components["text_score"],

            "mbti_score":
                components["mbti_score"],

            "location_score":
                components["location_score"],

            "weights":
                components["weights"]
        })


    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return results[
        :number_of_matches
    ]


# ============================================================
# SAVE FEEDBACK
# ============================================================

def save_feedback(
    user_id,
    matched_user_id,
    action
):

    action_value = (
        1
        if action == "accept"
        else 0
    )


    new_feedback = pd.DataFrame([
        {

            "user_id":
                str(user_id),

            "matched_user_id":
                str(matched_user_id),

            "action":
                action_value,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    ])


    # --------------------------------------------------------
    # Remove previous decision for same pair
    # --------------------------------------------------------

    if FEEDBACK_FILE.exists():

        current_feedback = pd.read_csv(
            FEEDBACK_FILE
        )

        if not current_feedback.empty:

            current_feedback[
                "user_id"
            ] = (
                current_feedback[
                    "user_id"
                ].astype(str)
            )

            current_feedback[
                "matched_user_id"
            ] = (
                current_feedback[
                    "matched_user_id"
                ].astype(str)
            )


            current_feedback = (
                current_feedback[
                    ~(
                        (
                            current_feedback[
                                "user_id"
                            ]
                            == str(user_id)
                        )
                        &
                        (
                            current_feedback[
                                "matched_user_id"
                            ]
                            == str(
                                matched_user_id
                            )
                        )
                    ]
                ]
            )

        else:

            current_feedback = pd.DataFrame(
                columns=[
                    "user_id",
                    "matched_user_id",
                    "action",
                    "timestamp"
                ]
            )

    else:

        current_feedback = pd.DataFrame(
            columns=[
                "user_id",
                "matched_user_id",
                "action",
                "timestamp"
            ]
        )


    # --------------------------------------------------------
    # Add latest feedback
    # --------------------------------------------------------

    current_feedback = pd.concat(
        [
            current_feedback,
            new_feedback
        ],
        ignore_index=True
    )


    current_feedback.to_csv(
        FEEDBACK_FILE,
        index=False
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🤖 AI Profile Matching System"
)

st.write(
    "An adaptive profile matching system that "
    "learns from user Accept/Reject feedback."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧠 AI Model"
    )


    st.write(
        f"👥 Users: **{len(users)}**"
    )


    st.write(
        f"📝 Feedback records: "
        f"**{len(dynamic_feedback_features)}**"
    )


    st.subheader(
        "Current Weights"
    )


    st.write(
        f"Profile Similarity: "
        f"**{dynamic_global_weights[0] * 100:.2f}%**"
    )


    st.write(
        f"MBTI Compatibility: "
        f"**{dynamic_global_weights[1] * 100:.2f}%**"
    )


    st.write(
        f"Location Compatibility: "
        f"**{dynamic_global_weights[2] * 100:.2f}%**"
    )


    if len(dynamic_user_weights) > 0:

        st.success(
            f"Personalized models: "
            f"{len(dynamic_user_weights)} users"
        )

    else:

        st.info(
            "Personalized learning becomes "
            "available after enough mixed feedback "
            "from a user."
        )


# ============================================================
# USER SELECTION
# ============================================================

st.subheader(
    "👤 Select Your Profile"
)


user_options = users[
    "user_id"
].tolist()


selected_user = st.selectbox(

    "Choose a user",

    user_options,

    format_func=lambda user_id: (

        f"{users.loc["
        "users['user_id'] == user_id,"
        "'name'"
        "].iloc[0]} "
        f"({user_id})"
    )
)


selected_profile = users[
    users["user_id"]
    == selected_user
].iloc[0]


# ============================================================
# USER PROFILE DISPLAY
# ============================================================

with st.expander(
    "View Selected Profile",
    expanded=True
):

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            f"**Name:** "
            f"{selected_profile['name']}"
        )

        st.write(
            f"**Age:** "
            f"{selected_profile['age']}"
        )

        st.write(
            f"**Location:** "
            f"{selected_profile['location']}"
        )


    with col2:

        st.write(
            f"**Profession:** "
            f"{selected_profile['profession']}"
        )

        st.write(
            f"**Experience:** "
            f"{selected_profile['experience_years']} years"
        )


    with col3:

        st.write(
            f"**MBTI:** "
            f"{selected_profile['mbti']}"
        )

        st.write(
            f"**Interests:** "
            f"{selected_profile['intrests']}"
        )


    st.write(
        f"**Professional Summary:** "
        f"{selected_profile['professional_summary']}"
    )


    st.write(
        f"**About Me:** "
        f"{selected_profile['about_me']}"
    )


# ============================================================
# NUMBER OF MATCHES
# ============================================================

max_matches = min(
    10,
    max(1, len(users) - 1)
)


number_of_matches = st.slider(

    "Number of matches",

    min_value=1,

    max_value=max_matches,

    value=min(5, max_matches)
)


# ============================================================
# FIND MATCHES
# ============================================================

if st.button(
    "🔍 Find Top Matches",
    type="primary",
    use_container_width=True
):

    st.session_state.matches = (
        get_top_matches(
            selected_user,
            number_of_matches
        )
    )


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "matches" not in st.session_state:

    st.session_state.matches = None


# ============================================================
# DISPLAY MATCHES
# ============================================================

if st.session_state.matches is not None:

    st.subheader(
        "🎯 Recommended Profiles"
    )


    matches = st.session_state.matches


    if len(matches) == 0:

        st.warning(
            "No matches found."
        )


    for rank, row in enumerate(
        matches,
        start=1
    ):

        st.markdown("---")


        # ----------------------------------------------------
        # MATCH HEADER
        # ----------------------------------------------------

        st.subheader(
            f"#{rank} — {row['name']}"
        )


        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.write(
                f"📍 **Location:** "
                f"{row['location']}"
            )

            st.write(
                f"💼 **Profession:** "
                f"{row['profession']}"
            )


        with col2:

            st.write(
                f"🧠 **MBTI:** "
                f"{row['mbti']}"
            )

            st.write(
                f"⭐ **Experience:** "
                f"{row['experience_years']} years"
            )


        with col3:

            st.metric(
                "Compatibility",
                f"{row['score']:.2f}%"
            )


        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        st.write(
            f"**Professional Summary:** "
            f"{row['professional_summary']}"
        )


        st.write(
            f"**About:** "
            f"{row['about_me']}"
        )


        st.write(
            f"**Interests:** "
            f"{row['intrests']}"
        )


        # ----------------------------------------------------
        # MATCH ANALYSIS
        # ----------------------------------------------------

        st.write(
            "### 📊 Match Analysis"
        )


        score1, score2, score3 = st.columns(3)


        with score1:

            st.metric(
                "Profile Similarity",
                f"{row['text_score']:.2f}%"
            )


        with score2:

            st.metric(
                "MBTI Compatibility",
                f"{row['mbti_score']:.2f}%"
            )


        with score3:

            st.metric(
                "Location Compatibility",
                f"{row['location_score']:.2f}%"
            )


        # ----------------------------------------------------
        # WHY THIS MATCH
        # ----------------------------------------------------

        reasons = []


        if row["text_score"] >= 50:

            reasons.append(
                "Strong similarity in profile "
                "content, interests and profession."
            )

        elif row["text_score"] >= 25:

            reasons.append(
                "Moderate similarity in profile "
                "content and interests."
            )

        else:

            reasons.append(
                "Limited textual similarity."
            )


        if row["mbti_score"] >= 100:

            reasons.append(
                "Both users have the same MBTI type."
            )

        elif row["mbti_score"] >= 70:

            reasons.append(
                "Both users belong to the same "
                "MBTI personality group."
            )

        else:

            reasons.append(
                "Users belong to different MBTI groups."
            )


        if row["location_score"] >= 100:

            reasons.append(
                "Both users are from the same location."
            )

        else:

            reasons.append(
                "Users are from different locations."
            )


        st.write(
            "### 💡 Why this match?"
        )


        for reason in reasons:

            st.write(
                f"• {reason}"
            )


        # ----------------------------------------------------
        # ACCEPT / REJECT
        # ----------------------------------------------------

        st.write(
            "### 🤝 Your Feedback"
        )


        accept_col, reject_col = st.columns(2)


        with accept_col:

            if st.button(
                "👍 Accept",
                key=(
                    f"accept_"
                    f"{selected_user}_"
                    f"{row['user_id']}"
                ),
                use_container_width=True
            ):

                save_feedback(
                    selected_user,
                    row["user_id"],
                    "accept"
                )


                st.success(
                    f"Accepted {row['name']} ✅"
                )


                st.toast(
                    "Feedback saved!"
                )


                st.rerun()


        with reject_col:

            if st.button(
                "👎 Reject",
                key=(
                    f"reject_"
                    f"{selected_user}_"
                    f"{row['user_id']}"
                ),
                use_container_width=True
            ):

                save_feedback(
                    selected_user,
                    row["user_id"],
                    "reject"
                )


                st.warning(
                    f"Rejected {row['name']}."
                )


                st.toast(
                    "Feedback saved!"
                )


                st.rerun()


# ============================================================
# FEEDBACK INFORMATION
# ============================================================

st.markdown("---")


with st.expander(
    "📈 View Feedback History"
):

    if FEEDBACK_FILE.exists():

        current_feedback = pd.read_csv(
            FEEDBACK_FILE
        )


        if current_feedback.empty:

            st.info(
                "No feedback recorded yet."
            )

        else:

            st.dataframe(
                current_feedback,
                use_container_width=True
            )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "📋 View Users Dataset"
):

    display_columns = [
        column
        for column in REQUIRED_USER_COLUMNS
        if column in users.columns
    ]


    st.dataframe(
        users[display_columns],
        use_container_width=True
    )
