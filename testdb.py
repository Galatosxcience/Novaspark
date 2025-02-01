import pymongo
import logging
import streamlit as st
from together import Together
from PIL import Image
import requests
from io import BytesIO

# MongoDB Configuration
MONGO_URI = "mongodb+srv://rennythomas:renny123@cluster0.7qopb.mongodb.net/scraping?retryWrites=true&w=majority"
DB_NAME = "scraping"
COLLECTION_NAME = "phonescraper"

# Initialize Together client with your API key
api_key = "861814cbdeb732c7cd715952c62fde5db61b68d1f7f9fc8dedf6c04baef832f6"  # Replace with your actual API key
client = Together(api_key=api_key)

# MongoDB Helper Function to Fetch Data
def fetch_data_from_mongo():
    """Fetch relevant data from MongoDB collection"""
    try:
        client_db = pymongo.MongoClient(MONGO_URI)
        db = client_db[DB_NAME]
        collection = db[COLLECTION_NAME]
        data = list(collection.find({}, {"_id": 0}))  # Fetch data without _id field
        return data
    except Exception as e:
        logging.error(f"Error fetching data from MongoDB: {e}")
        return []

# Function to Filter Phones under 20,000 INR
def filter_phones_under_20k(data):
    """Filter phones based on price under 20,000 INR"""
    return [phone for phone in data if float(phone.get('price', '0').replace('₹', '').replace(',', '')) <= 20000]

# Function to Display Image if URL is Provided (with SSL verification disabled)
def display_image_from_document(image_url):
    """Extract and display image if available in the document."""
    if image_url:
        try:
            # Disable SSL verification for image fetching
            response = requests.get(image_url, verify=False)  # Bypasses SSL certificate validation
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Product Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

# Function to Display Phone Information
def display_phone_info(phone):
    """Display the relevant information and image of the phone."""
    st.write(f"### {phone['name']}")
    st.write(f"**Price**: {phone['price']}")
    st.write("**Specifications**:")
    for spec, value in phone.get('specifications', {}).items():
        st.write(f" - **{spec}**: {value}")
    
    # Display the image if the URL is available
    display_image_from_document(phone.get('image'))

# Function to Fetch Chat Response with RAG Model
def get_chat_response_with_rag(query):
    try:
        # Step 1: Fetch data from MongoDB
        data = fetch_data_from_mongo()

        # Step 2: Filter phones under 20,000 INR
        relevant_phones = filter_phones_under_20k(data)

        # If no phones are found, display a message
        if not relevant_phones:
            st.write("Sorry, no phones found under 20,000 INR based on your query.")
            return

        # Step 3: Show filtered phone details
        for phone in relevant_phones:
            display_phone_info(phone)

        # Step 4: Create context from the relevant phones for RAG
        context = " ".join([str(phone) for phone in relevant_phones[:3]])  # Using top 3 for simplicity

        # Step 5: Send the query and context to the model for completion
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[
                {
                    "role": "system",
                    "content": "<human>: " + query + " <bot>: I will assist you based on the following context: " + context
                }
            ],
            max_tokens=None,  # Set to None, not null
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<｜end▁of▁sentence｜>"],
            stream=True  # Enable streaming
        )

        # Handle the response in stream and return the result
        response_text = ""
        for token in response:
            if hasattr(token, 'choices'):
                response_text += token.choices[0].delta.content

        # Display the chatbot's response
        st.write(f"**SupportBot**: {response_text}")

    except Exception as e:
        logging.error(f"Error during chat with RAG request: {e}")
        st.write("Sorry, I encountered an error while processing your request.")

# Streamlit UI for user input and response display
def main():
    # Title of the app
    st.title("📱 Phone Information & Support Chatbot")

    # Display a text input box for user query
    user_input = st.text_input("🔍 Ask about the latest phones under 20,000 INR or any other query:")

    if user_input:
        # Display loading message while processing the query
        st.write("Processing your request... Please wait.")

        # Step 1: Get the response from RAG model and display phones
        get_chat_response_with_rag(user_input)

# Run the Streamlit app
if __name__ == "__main__":
    main()
