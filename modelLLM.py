import logging
import streamlit as st
import pymongo
import time
from together import Together

# Configure logging
logging.basicConfig(level=logging.INFO)

##############################################
# MongoDB Configuration from Secrets
##############################################
MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db_name"]
COLLECTION_NAME = st.secrets["mongo"]["collection_name"]

##############################################
# Together API Configuration from Secrets
##############################################
API_KEY = st.secrets["together"]["api_key"]
client = Together(api_key=API_KEY)

##############################################
# Model Functions Using Together AI API
##############################################
def load_model():
    return "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def generate_answer(prompt, model_name):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert phone advisor specialized in recommending phones."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
            top_p=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Together API error: {e}")
        return "I'm sorry, I encountered an issue generating the answer."

##############################################
# Data Retrieval and Processing Functions
##############################################
def fetch_data_from_mongo():
    try:
        client_db = pymongo.MongoClient(MONGO_URI)
        db = client_db[DB_NAME]
        collection = db[COLLECTION_NAME]
        data = list(collection.find({}, {"_id": 0}))
        logging.info(f"Fetched {len(data)} records from MongoDB.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data from MongoDB: {e}")
        return []

def remove_duplicates(data):
    seen = set()
    unique_data = []
    for phone in data:
        key = (phone.get('name'), phone.get('price'))
        if key not in seen:
            seen.add(key)
            unique_data.append(phone)
    return unique_data

##############################################
# Query Handling and Prompt Building Functions
##############################################
def is_phone_related(query):
    phone_keywords = ['phone', 'specifications', 'storage', 'ram', 'price', 'model', 'android', 'ios', 'gaming', 'camera', 'iphone']
    return any(keyword in query.lower() for keyword in phone_keywords)

def handle_special_queries(user_input):
    user_input_lower = user_input.lower()
    if 'name' in user_input_lower or 'who are you' in user_input_lower:
        return "My name is **Novaspark**, your intelligent phone advisor. How can I assist you today?"
    if 'created' in user_input_lower or 'who created you' in user_input_lower:
        return (
            "I was created by **Novaspark Pvt Limited**, with the following amazing founders:\n"
            "- Aljo Joseph\n"
            "- Aldon Alphonses Tom\n"
            "- Ashik Shaji\n"
            "- Rojin S. Martin\n"
            "- Renny Thomas"
        )
    if any(greeting in user_input_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon']):
        return "Hello! I’m **Novaspark**, your intelligent assistant. How can I help you today?"
    return None

def build_phone_prompt(user_query, data):
    # Build a dynamic prompt based on user query
    prompt = (
        f"You are an expert phone advisor. The user is asking about phones based on the following query:\n"
        f"User Query: {user_query}\n\n"
        "Please provide a recommendation for the best phones that match the query, considering the available phones in the database."
    )
    return prompt

##############################################
# Main Streamlit Application
##############################################
def main():
    st.set_page_config(page_title="Novaspark - Intelligent Phone Advisor", layout="wide")
    st.title("📱 Novaspark Intelligent Phone Advisor")

    # Main Input Box for the user to type their query
    user_input = st.text_input("🔍 Ask for phone recommendations or any query:")

    # If user has typed something
    if user_input:
        data = fetch_data_from_mongo()
        unique_phones = remove_duplicates(data)

        special_response = handle_special_queries(user_input)
        if special_response:
            st.write(special_response)
        elif not is_phone_related(user_input):
            st.write("Please ask a phone-related query.")
        else:
            # Pass the query and the available phones to the model
            prompt = build_phone_prompt(user_input, unique_phones)
            model_name = load_model()
            answer = generate_answer(prompt, model_name)
            st.write(answer)

if __name__ == "__main__":
    main()
