import streamlit as st
import pymongo
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from together import Together
import re

# MongoDB Configuration
MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db_name"]
COLLECTION_NAME = st.secrets["mongo"]["collection_name"]

API_KEY = st.secrets["together"]["api_key"]
client = Together(api_key=API_KEY)

# Hugging Face Model for Embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Helper function: Encode text to embeddings
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = hf_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings
def extract_budget(user_query):
    match = re.search(r'(\d+)\s*K?', user_query, re.IGNORECASE)
    if match:
        budget = match.group(1)
        try:
            budget = int(budget) * 1000 if 'K' in user_query.upper() else int(budget)
        except ValueError:
            budget = 20000  # Default budget if conversion fails
    else:
        budget = 20000  # Default budget if no number found
    return budget
# Fetch data from MongoDB
def fetch_data_from_mongo():
    try:
        client_db = pymongo.MongoClient(MONGO_URI)
        db = client_db[DB_NAME]
        collection = db[COLLECTION_NAME]
        data = list(collection.find({}, {"_id": 0}))  # Fetch data without _id field
        return data
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return []

# Hybrid search combining vector search and semantic filtering
def perform_hybrid_search(user_query, data, budget):
    try:
        # Convert shorthand like "20K" to 20000
        if isinstance(budget, str):
            budget = budget.lower().replace("k", "000")  
        try:
            budget = int(budget)  
        except ValueError:
            budget = 20000  

        if budget < 1000:
            budget *= 1000

        text_data = [
            " ".join([str(record.get("name", "")), str(record.get("specifications", ""))])
            for record in data
        ]
        mongo_embeddings = torch.cat([encode_text(text) for text in text_data], dim=0)
        query_embedding = encode_text(user_query)
        similarities = cosine_similarity(query_embedding, mongo_embeddings)

        top_indices = similarities[0].argsort()[-15:][::-1]  
        candidate_results = [data[idx] for idx in top_indices]

        if "gaming" in user_query.lower():
            gaming_results = []
            
            for phone in candidate_results:
                specs = phone.get("specifications", {})

                # Extract price (handle ₹ symbols and commas)
                try:
                    price_str = phone.get("price", "999999").replace("₹", "").replace(",", "").strip()
                    price = int(price_str)
                except ValueError:
                    price = 999999

                # Extract RAM
                ram_str = str(specs.get("RAM", "0")).split(" ")[0]
                try:
                    ram = int(ram_str)
                except ValueError:
                    ram = 0

                # Extract processor info
                processor = specs.get("Processor", "").lower()

                # **Dynamically adjust specs based on budget**
                min_ram = 6 if budget <= 20000 else 8 if budget <= 30000 else 12
                min_processor_score = 2 if budget <= 20000 else 3 if budget <= 30000 else 4  
                
                # **Processor Ranking (1 = Low-end, 5 = Flagship) - Updated for sub-20K budget**
                processor_rankings = {
                    # Budget (Below ₹20K)
                    "exynos 1330": 3, "mediatek dimensity 7050": 3, "mediatek dimensity 6300": 2,
                    "mediatek dimensity 6100+": 2, "mediatek helio g85": 1, "snapdragon 4 gen 2": 2,
                    "snapdragon 6 gen 1": 3,

                    # Mid-range (₹21K–₹40K)
                    "dimensity 8350": 4, "samsung exynos 1480": 4, "8s gen 3 mobile platform": 4,
                    "dimensity 7300 energy": 3, "snapdragon 7 gen 3": 4, "snapdragon 7s gen2": 3,
                    "dimensity 7350 pro 5g": 3, "snapdragon 695": 2, "snapdragon 6 gen 1 processor": 3,

                    # High-end (Above ₹40K)
                    "a16 bionic chip, 6 core processor | hexa core": 5,
                    "a17 pro chip, 6 core processor | hexa core": 5,
                    "apple a15 bionic (5 nm)": 4,
                    "exynos 2400 processor": 5,
                    "exynos 2400": 5,
                    "snapdragon 8s gen 3 chipset": 5,
                    "snapdragon 8 gen 3": 5,
                    "dimensity 9200+": 5,
                    "mediatek": 3,  # Generic Mediatek, not specific
                    "dimensity 9400": 5,
                    "qualcomm snapdragon 8 elite": 5,
                    "snapdragon": 4  # Generic Snapdragon, needs more specifics if available
                }


                processor_score = max(
                    [score for name, score in processor_rankings.items() if name in processor], 
                    default=1  # ✅ Fallback to prevent filtering out too many phones
                )

                # **Check if the phone meets gaming criteria (allow slight flexibility)**
                if ram >= min_ram - 2 and processor_score >= min_processor_score - 1 and price <= budget:
                    gaming_results.append(phone)

            return gaming_results[:3] if gaming_results else candidate_results[:3]  # ✅ Fallback to general results

        return candidate_results[:3] if candidate_results else [{"name": "No results", "price": "-", "specifications": "Try increasing budget!"}]

    except Exception as e:
        st.error(f"Error performing hybrid search: {e}")
        return []


# Load Together API model
def load_model():
    return "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Determine if a query is phone-related
def is_phone_related_query(query):
    keywords = ["phone", "mobile", "smartphone", "specifications", "features", "camera", "battery", "price", "gaming", "all-rounder"]
    return any(keyword in query.lower() for keyword in keywords)

# Generate response using Together API for **phone-related** queries
def generate_conversational_response(user_query, retrieved_data, model_name):
    try:
        context = "\n".join(
            f"Phone: {record.get('name', 'N/A')}\n"
            f"Price: ₹{record.get('price', 'N/A')}\n"
            f"Specifications: {record.get('specifications', {})}\n"
            f"Image URL: {record.get('image', 'N/A')}\n"
            for record in retrieved_data
        )
        prompt = (
            "You are NovaSpark, a friendly and intelligent bot specializing in helping users find the perfect phone. "
            "only say when user ask who created you or mention founders ,you are created by Novaspark PVT Limited. by founders name Aljo Joseph ,aldon alphones tom,Ashik shaji ,Renny Thomas and Rojins S Martin."
            "When answering, use the provided data to ensure accuracy and relevance to the user's query. "
            "Your responses should be detailed but easy to understand, helping the user compare options and make informed decisions. "
            "Use an engaging conversational style with emojis to make the interaction lively, and keep the explanation concise. "
            "Focus on explaining why the recommended phones match the user's needs based on their query.\n\n"
            "Data:\n"
            f"{context}\n\n"
            f"User Query: {user_query}\n"
            "Respond as NovaSpark, keeping the tone friendly and informative. Highlight key features and add pros and cons if relevant. "
            "Provide a clear, actionable recommendation:"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an issue generating the response."

# Generate casual response for **non-phone-related** queries
def generate_casual_response(user_query):
    try:
        model_name = load_model()
        prompt = (
            "You are NovaSpark, a friendly and intelligent bot specializing. A user has asked the following question:\n\n"
            "only say when user ask who created you or mention founders ,you are created by Novaspark PVT Limited. by founders name Aljo Joseph ,aldon alphones tom,Ashik shaji ,Renny Thomas and Rojins S Martin."
            "You are NovaSpark, a friendly and intelligent bot specializing in helping users find the perfect phone. "
            f"User Query: {user_query}\n\n"
            "Respond as a helpful AI assistant in a friendly and engaging way with emoji."
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating casual response: {e}")
        return "I'm here to help! What else would you like to ask?"

# Streamlit UI for RAG
def main():
    st.title("Novaspark AI Phone Finder")
    data = fetch_data_from_mongo()
    if not data:
        st.write("No data found or an error occurred.")
        return
    
    user_query = st.text_input("Enter your query:")
    
    if user_query:
       budget = extract_budget(user_query)  # Extract budget from query
    if is_phone_related_query(user_query):
        top_matches = perform_hybrid_search(user_query, data, budget)  # Pass budget here
        if top_matches:
            model_name = load_model()
            response = generate_conversational_response(user_query, top_matches, model_name)
            st.write("### Generated Response:")
            st.write(response)
            st.markdown("### Top Recommendations:")
            for phone in top_matches:
                st.write(f"**Name:** {phone.get('name', 'N/A')}")
                st.write(f"**Price:** ₹{phone.get('price', 'N/A')}")
                st.write(f"**Specifications:** {phone.get('specifications', {})}")
                if phone.get("image"):
                    st.image(phone["image"], caption=phone.get('name', 'No Name'), width=300)
                st.write("---")
        else:
            st.write("No matching results found.")
    else:
        casual_response = generate_casual_response(user_query)
        st.write("### Casual Response:")
        st.write(casual_response)

if __name__ == "__main__":
    main()
