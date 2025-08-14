import os
import json
import time
import requests
import streamlit as st

# This is a chatbot with a Streamlit UI and analytics.
# It communicates with the Gemini API and logs interactions.

# --- Configuration ---
# You can set your API key here, or as a Streamlit secret.
# In production, using secrets is highly recommended.
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("API key not found. Please set the GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
ANALYTICS_FILE = "analytics.jsonl"

# --- Chatbot Core Functions ---

def exponential_backoff_fetch(url, payload, retries=5, delay=1):
    """Handles API calls with exponential backoff for robustness."""
    full_url = f"{url}?key={API_KEY}"
    for i in range(retries):
        try:
            response = requests.post(full_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if i < retries - 1:
                st.error(f"Request failed, retrying in {delay} seconds... ({e})")
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"An error occurred after multiple retries: {e}")
                raise
    return None

def get_gemini_response(prompt):
    """Sends a prompt to the Gemini API and returns the text response."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    try:
        response_data = exponential_backoff_fetch(API_URL, payload)
        if response_data and response_data.get('candidates'):
            return response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Sorry, I could not get a valid response from the API."
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return f"An internal error occurred: {e}"

def classify_topic(query):
    """Uses the Gemini API to classify the topic of a user's query."""
    topic_prompt = (
        f"Classify the following text into a single, concise topic (e.g., "
        f"'General Question', 'Technical Support', 'Product Information', 'Creative Writing'). "
        f"Text: '{query}'"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": topic_prompt}]}]
    }

    try:
        response_data = exponential_backoff_fetch(API_URL, payload)
        if response_data and response_data.get('candidates'):
            return response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return "Unknown"
    except Exception as e:
        st.error(f"An error occurred during topic classification: {e}")
        return "Unknown"

# --- Analytics Functions ---

def log_analytics(data):
    """Appends a new interaction log to the analytics file."""
    try:
        with open(ANALYTICS_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except IOError as e:
        st.error(f"Error writing to analytics file: {e}")

def get_analytics_dashboard_data():
    """Reads the analytics file and returns a summary."""
    total_queries = 0
    topic_counts = {}
    positive_ratings = 0
    negative_ratings = 0
    
    try:
        with open(ANALYTICS_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_queries += 1
                    if data.get('rating') == 'positive':
                        positive_ratings += 1
                    elif data.get('rating') == 'negative':
                        negative_ratings += 1
                    
                    topic = data.get('topic', 'Unknown')
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
        
    sorted_topics = sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "totalQueries": total_queries,
        "positiveRatings": positive_ratings,
        "negativeRatings": negative_ratings,
        "topics": sorted_topics
    }

# --- Streamlit UI ---

st.title("Chatbot with Analytics Dashboard")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for the dashboard
with st.sidebar:
    st.header("Analytics Dashboard")
    analytics_data = get_analytics_dashboard_data()
    st.metric("Total Queries", analytics_data["totalQueries"])
    st.metric("Positive Ratings", analytics_data["positiveRatings"])
    st.metric("Negative Ratings", analytics_data["negativeRatings"])
    st.subheader("Most Common Topics")
    for topic, count in analytics_data["topics"]:
        st.write(f"- **{topic}**: {count} times")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        # Get bot response and classify topic
        bot_response = get_gemini_response(prompt)
        topic = classify_topic(prompt)
    
    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
        # Add rating buttons
        if bot_response:
            col1, col2 = st.columns([1, 1])
            if col1.button("👍 Helpful", key="helpful"):
                log_analytics({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "query": prompt, "response": bot_response, "topic": topic, "rating": "positive"})
                st.info("Thanks for your feedback!")
            if col2.button("👎 Not Helpful", key="not-helpful"):
                log_analytics({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "query": prompt, "response": bot_response, "topic": topic, "rating": "negative"})
                st.info("Thanks for your feedback!")

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
