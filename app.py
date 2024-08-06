import threading
import time
import requests
import streamlit as st
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def keep_awake(url, interval):
    while True:
        try:
            requests.get(url)
        except Exception as e:
            print(f"Error keeping app awake: {e}")
        time.sleep(interval)

# URL of your Streamlit app (use your actual app URL)
app_url = "https://genai-project-7jrzz8uukutamsphsvzwcw.streamlit.app/"

# Start the thread to keep the app awake
threading.Thread(target=keep_awake, args=(app_url, 300)).start()

# Add the rest of your app content here

# Load precomputed embeddings
df_with_embeddings = pd.read_pickle('df_with_embeddings.pkl')

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
        st.write(f"*Package Name:* {selected_row['package_name'].values[0]}")
        st.write(f"*Itinerary:* {selected_row['itinerary'].values[0]}")
        st.write(f"*Sightseeing Places Covered:* {selected_row['sightseeing_places_covered'].values[0]}")
    else:
        st.write("Invalid selection. No package found.")

def evaluate_model(df, model):
    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Encode the sentences
    train_embeddings = model.encode(train_df['description'].tolist(), convert_to_tensor=True)
    test_embeddings = model.encode(test_df['description'].tolist(), convert_to_tensor=True)

    # Function to get the most similar label from the training set for a given test embedding
    def get_most_similar_label(test_embedding, train_embeddings, train_labels):
        similarities = util.pytorch_cos_sim(test_embedding, train_embeddings)
        most_similar_idx = similarities.argmax().item()
        return train_labels[most_similar_idx]

    # Predict labels for the test set
    predicted_labels = [get_most_similar_label(embed, train_embeddings, train_df['Primary'].tolist()) for embed in test_embeddings]

    # Calculate accuracy metrics
    accuracy = accuracy_score(test_df['Primary'], predicted_labels)
    precision = precision_score(test_df['Primary'], predicted_labels, average='weighted')
    recall = recall_score(test_df['Primary'], predicted_labels, average='weighted')
    f1 = f1_score(test_df['Primary'], predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

# Streamlit app
st.title("Travel Recommendation System")

st.write("Please provide your travel preferences below:")

user_input = get_user_input()

if st.button("Get Recommendations"):
    recommendations = recommend_destinations(user_input, df_with_embeddings)
    st.write("Top recommended destinations for you:")
    st.session_state.recommendations = recommendations
    st.dataframe(recommendations)

if 'recommendations' in st.session_state:
    primary_selection = st.selectbox("Select a package to view details", options=st.session_state.recommendations['Primary'].tolist())
    if st.button("View Details"):
        st.session_state.selected_package = primary_selection

if 'selected_package' in st.session_state:
    st.write(f"Details for {st.session_state.selected_package}:")
    display_package_details(st.session_state.selected_package, df_with_embeddings)

if st.button("Evaluate Model Accuracy"):
    accuracy, precision, recall, f1 = evaluate_model(df_with_embeddings, model)
    st.write(f'Accuracy: {accuracy}')
    st.write(f'Precision: {precision}')
    st.write(f'Recall: {recall}')
    st.write(f'F1 Score: {f1}')




