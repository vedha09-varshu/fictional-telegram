from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import random
import streamlit as st

# Generate synthetic data
data = {
    "Openness": [random.randint(1, 5) for _ in range(100)],
    "Conscientiousness": [random.randint(1, 5) for _ in range(100)],
    "Extraversion": [random.randint(1, 5) for _ in range(100)],
    "Agreeableness": [random.randint(1, 5) for _ in range(100)],
    "Neuroticism": [random.randint(1, 5) for _ in range(100)],
}

# Map personality traits to color palettes
def get_color_palette(openness, conscientiousness, extraversion, agreeableness, neuroticism):
    if extraversion > 3:
        return "Warm Colors (Red, Orange)"
    elif openness > 4:
        return "Creative Colors (Purple, Magenta)"
    elif neuroticism > 3:
        return "Neutral Colors (Gray, Black)"
    elif agreeableness > 4:
        return "Soft Colors (Yellow, Pink)"
    else:
        return "Cool Colors (Blue, Green)"

data["Color_Palette"] = [
    get_color_palette(row["Openness"], row["Conscientiousness"], row["Extraversion"], row["Agreeableness"], row["Neuroticism"])
    for _, row in pd.DataFrame(data).iterrows()
]

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("personality_color_data.csv", index=False)

# Load the dataset
data = pd.read_csv("personality_color_data.csv")
X = data.drop("Color_Palette", axis=1)
y = data["Color_Palette"]

# Encode the target variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(max_features='sqrt')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Streamlit App
st.title("Personality-Based Color Palette Recommender")
st.write("Answer the questions below to get a personalized color palette recommendation!")

# Pre-defined questions
questions = [
    "I enjoy large social gatherings.",
    "I often try new and unusual activities.",
    "I am detail-oriented and organized.",
    "I often feel anxious or stressed.",
    "I trust people easily."
]

# User responses
responses = []

for i, question in enumerate(questions):
    response = st.radio(question, ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], key=f"q{i}")
    responses.append(response)

# Map responses to numerical values
response_mapping = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}
numerical_responses = [response_mapping[r] for r in responses]

# Predict personality and recommend a color palette
if st.button("Get Color Palette"):
    try:
        predicted_index = model.predict([numerical_responses])[0]
        predicted_palette = encoder.inverse_transform([predicted_index])[0]
        st.write(f"Your recommended color palette is: {predicted_palette}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
