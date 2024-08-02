import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import gdown
import os

# Google Drive URL of the pickle file
url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
output = 'df_with_embeddings.pkl'

# Download the pickle file from Google Drive
try:
    gdown.download(url, output, quiet=False)
    if os.path.exists(output):
        st.write(f"File downloaded successfully: {output}")
        st.write(f"File size: {os.path.getsize(output)} bytes")
    else:
        st.error(f"Failed to download the file: {output}")
except Exception as e:
    st.error(f"An error occurred while downloading the file: {e}")

# Load precomputed embeddings
try:
    with open(output, 'rb') as f:
        df_with_embeddings = pd.read_pickle(f)
    st.write("Pickle file loaded successfully.")
except Exception as e:
    st.error(f"An error occurred while loading the pickle file: {e}")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_user_input():
    companions = st.selectbox("Who are you traveling with?", options=["solo", "couple", "family"])

    if companions == "solo":
        num_people = 1
    elif companions == "couple":
        num_people = 2
    elif companions == "family":
        num_people = st.number_input("Enter the number of people:", min_value=1, step=1)
    
    budget = st.number_input("Enter your budget per person:", min_value=0.0, step=0.01)
    days_of_lodging = st.number_input("Enter the number of days of lodging:", min_value=1, step=1)
    preferred_weather = st.selectbox("Enter preferred weather:", options=["Sunny", "Rainy", "Snowy"])

    return budget, num_people, companions, days_of_lodging, preferred_weather

def encode_user_input(user_input):
    user_description = f"budget {user_input[0]} companions {user_input[2]} days {user_input[3]} weather {user_input[4]}"
    return model.encode(user_description, convert_to_tensor=True)

def recommend_destinations(user_input, df):
    user_embedding = encode_user_input(user_input)
    df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(user_embedding, x).item())

    # Sort by similarity
    recommendations = df.sort_values(by='similarity', ascending=False).drop_duplicates(subset='Primary').head(5)

    return recommendations[['Primary', 'per_person_price', 'Topography', 'Temprature', 'Weather', 'Mood']]

def display_package_details(selection, df):
    selected_row = df.loc[df['Primary'] == selection]
    if not selected_row.empty:
        st.write(f"**Package Name:** {selected_row['Primary'].values[0]}")
        st.write(f"**Price per person:** {selected_row['per_person_price'].values[0]}")
        st.write(f"**Topography:** {selected_row['Topography'].values[0]}")
        st.write(f"**Temperature:** {selected_row['Temprature'].values[0]}")
        st.write(f"**Weather:** {selected_row['Weather'].values[0]}")
        st.write(f"**Mood:** {selected_row['Mood'].values[0]}")

def evaluate_model(df, model):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    train_embeddings = list(df_train['embedding'])
    test_embeddings = list(df_test['embedding'])

    def get_most_similar_label(test_embedding, train_embeddings, train_labels):
        similarities = util.pytorch_cos_sim(test_embedding, train_embeddings)
        most_similar_idx = similarities.argmax().item()
        return train_labels[most_similar_idx]

    # Predict labels for the test set
    predicted_labels = [get_most_similar_label(embed, train_embeddings, df_train['Primary'].tolist()) for embed in test_embeddings]

    # Calculate accuracy metrics
    accuracy = accuracy_score(df_test['Primary'], predicted_labels)
    precision = precision_score(df_test['Primary'], predicted_labels, average='weighted')
    recall = recall_score(df_test['Primary'], predicted_labels, average='weighted')
    f1 = f1_score(df_test['Primary'], predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

# Streamlit app
st.title("Travel Recommendation System")

st.write("Please provide your travel preferences below:")

user_input = get_user_input()

if st.button("Get Recommendations"):
    if 'df_with_embeddings' in globals():
        recommendations = recommend_destinations(user_input, df_with_embeddings)
        st.write("Top recommended destinations for you:")
        st.session_state.recommendations = recommendations
        st.dataframe(recommendations)
    else:
        st.error("Data is not loaded properly.")

if 'recommendations' in st.session_state:
    primary_selection = st.selectbox("Select a package to view details", options=st.session_state.recommendations['Primary'].tolist())
    if st.button("View Details"):
        st.session_state.selected_package = primary_selection

if 'selected_package' in st.session_state:
    st.write(f"Details for {st.session_state.selected_package}:")
    display_package_details(st.session_state.selected_package, df_with_embeddings)

if st.button("Evaluate Model Accuracy"):
    if 'df_with_embeddings' in globals():
        accuracy, precision, recall, f1 = evaluate_model(df_with_embeddings, model)
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1 Score: {f1}')
    else:
        st.error("Data is not loaded properly.")



