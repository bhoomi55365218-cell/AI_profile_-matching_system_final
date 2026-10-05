
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Profile Matching System",
    page_icon="🤝",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
users = pd.read_csv("/content/users.csv")
feedback = pd.read_csv("/content/feedback.csv")

# -----------------------------
# TEXT PROCESSING
# -----------------------------
users["clean_text"] = (
    users["professional_summary"].fillna("") + " " +
    users["about_me"].fillna("") + " " +
    users["intrests"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(users["clean_text"])

text_similarity = cosine_similarity(tfidf_matrix)

# -----------------------------
# MBTI COMPATIBILITY
# -----------------------------
mbti_groups = {
    "NT": ["INTJ", "INTP", "ENTJ", "ENTP"],
    "NF": ["INFJ", "INFP", "ENFJ", "ENFP"],
    "SJ": ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
    "SP": ["ISTP", "ISFP", "ESTP", "ESFP"]
}

def get_mbti_group(mbti):
    for group, types in mbti_groups.items():
        if mbti in types:
            return group
    return None


def mbti_compatibility(mbti1, mbti2):

    if mbti1 == mbti2:
        return 1.0

    if get_mbti_group(mbti1) == get_mbti_group(mbti2):
        return 0.7

    return 0.4


# -----------------------------
# LOCATION COMPATIBILITY
# -----------------------------
def location_compatibility(location1, location2):

    if location1 == location2:
        return 1.0

    return 0.0


# -----------------------------
# PREPARE FEEDBACK DATA
# -----------------------------
feedback_data = []

for _, row in feedback.iterrows():

    user_matches = users.index[
        users["user_id"] == row["user_id"]
    ]

    matched_matches = users.index[
        users["user_id"] == row["matched_user_id"]
    ]

    if len(user_matches) == 0 or len(matched_matches) == 0:
        continue

    user_index = user_matches[0]
    matched_index = matched_matches[0]

    text_score = text_similarity[
        user_index
    ][matched_index]

    mbti_score = mbti_compatibility(
        users.iloc[user_index]["mbti"],
        users.iloc[matched_index]["mbti"]
    )

    location_score = location_compatibility(
        users.iloc[user_index]["location"],
        users.iloc[matched_index]["location"]
    )

    feedback_data.append({
        "text_score": text_score,
        "mbti_score": mbti_score,
        "location_score": location_score,
        "action": row["action"]
    })


feedback_df = pd.DataFrame(feedback_data)

# -----------------------------
# TRAIN FEEDBACK MODEL
# -----------------------------
X = feedback_df[
    ["text_score", "mbti_score", "location_score"]
]

y = feedback_df["action"]

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

model = LogisticRegression()

model.fit(
    X_train_scaled,
    y_train
)

# -----------------------------
# LEARNED WEIGHTS
# -----------------------------
coefficients = model.coef_[0]

weights = abs(coefficients) / abs(coefficients).sum()

learned_w1 = weights[0]
learned_w2 = weights[1]
learned_w3 = weights[2]


# -----------------------------
# LEARNED MATCH SCORE
# -----------------------------
def calculate_learned_match_score(
    user_index,
    matched_index
):

    text_score = text_similarity[
        user_index
    ][matched_index]

    mbti_score = mbti_compatibility(
        users.iloc[user_index]["mbti"],
        users.iloc[matched_index]["mbti"]
    )

    location_score = location_compatibility(
        users.iloc[user_index]["location"]
         users.iloc[matched_indx]["location"])
