import logging
import streamlit as st
import pymongo
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
    if 'name' in user_input.lower() or 'who are you' in user_input.lower():
        return "My name is **Novaspark**, created by **Novaspark Pvt Limited**. How can I assist you today?"
    if 'created' in user_input.lower() or 'who created you' in user_input.lower():
        return (
            "I was created by **Novaspark Pvt Limited**, including these amazing team members:\n"
            "- Ashik Shaji (shajipapen)\n"
            "- Rojins Kannur\n"
            "- Aljo (Gymen)\n"
            "- Aldon Alphonese Tom (Magadi)"
        )
    return None

def filter_phones_by_query(user_input, data):
    if not data:
        return []
    filtered_phones = []
    if 'gaming' in user_input.lower():
        filtered_phones = [phone for phone in data if 'gaming' in str(phone.get('specifications', '')).lower() and float(phone.get('price', 0)) < 20000]
    elif 'camera' in user_input.lower():
        filtered_phones = [phone for phone in data if 'camera' in str(phone.get('specifications', '')).lower()]
    elif 'iphone' in user_input.lower():
        filtered_phones = [phone for phone in data if 'iphone' in phone.get('name', '').lower()]
    else:
        filtered_phones = data
    return filtered_phones[:3]

def build_phone_prompt(user_query, filtered_phones):
    context = "\n".join(
        f"Phone Name: {phone.get('name', 'N/A')}\n"
        f"Price: ₹{phone.get('price', 'N/A')}\n"
        f"Specifications: {phone.get('specifications', {})}\n"
        for phone in filtered_phones
    )
    prompt = (
        "You are an expert phone advisor. Based on the following phone details:\n"
        f"{context}\n\n"
        "And given the user's query below, provide a thoughtful recommendation.\n"
        f"User Query: {user_query}\nAnswer:"
    )
    return prompt

##############################################
# Main Streamlit Application
##############################################
def main():
    st.set_page_config(page_title="Novaspark - Intelligent Phone Advisor", layout="wide")
    st.title("📱 Novaspark Intelligent Phone Advisor")
    
    data = fetch_data_from_mongo()
    unique_phones = remove_duplicates(data)

    user_input = st.text_input("🔍 Ask for phone recommendations or any query:")
    
    if user_input:
        special_response = handle_special_queries(user_input)
        if special_response:
            st.write(special_response)
        elif not is_phone_related(user_input):
            st.write("Please ask a phone-related query.")
        else:
            filtered_phones = filter_phones_by_query(user_input, unique_phones)
            prompt = build_phone_prompt(user_input, filtered_phones)
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
