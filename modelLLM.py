import logging
import streamlit as st
import pymongo
import re  # To extract budget dynamically
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
            max_tokens=500,
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

def extract_budget(user_input):
    """
    Extracts the budget from the user input using regular expressions.
    The expected format is "under X" where X is a number (e.g., "under 20K", "under 80K").
    """
    match = re.search(r'under\s*(\d+)(?:k|K)', user_input)
    if match:
        return int(match.group(1)) * 1000  # Convert to full number (e.g., 20K -> 20000)
    return None

def filter_phones_by_query(user_input, data):
    if not data:
        return []
    
    # Extract budget from user input
    budget = extract_budget(user_input)
    if not budget:
        budget = 20000  # Default to ₹20K if no budget is mentioned
    
    filtered_phones = []
    
    # Filter phones based on user query keywords
    if 'gaming' in user_input.lower():
        filtered_phones = [
            phone for phone in data 
            if 'gaming' in str(phone.get('specifications', '')).lower() and float(phone.get('price', 0)) <= budget
        ]
    elif 'camera' in user_input.lower():
        filtered_phones = [
            phone for phone in data 
            if 'camera' in str(phone.get('specifications', '')).lower() and float(phone.get('price', 0)) <= budget
        ]
    else:
        filtered_phones = [
            phone for phone in data 
            if float(phone.get('price', 0)) <= budget
        ]
    
    # Return only top 3 recommendations
    return filtered_phones[:3]

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
            filtered_phones = filter_phones_by_query(user_input, unique_phones)
            prompt = f"User query: {user_input}.\n\n" + f"Filtered phones (up to {extract_budget(user_input)}):"
            model_name = load_model()
            answer = generate_answer(prompt, model_name)
            st.write(answer)

            if filtered_phones:
                st.markdown("### Top Recommendations:")
                for phone in filtered_phones:
                    st.write(f"**Name:** {phone.get('name', 'N/A')}")
                    st.write(f"**Price:** ₹{phone.get('price', 'N/A')}")
                    st.write(f"**Specifications:** {phone.get('specifications', {})}")
                    if phone.get("image"):
                        st.image(phone["image"], use_container_width=True)
                    st.write("---")

if __name__ == "__main__":
    main()
