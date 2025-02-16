import streamlit as st
import pymongo
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from together import Together
import re

MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db_name"]
COLLECTION_NAME = st.secrets["mongo"]["collection_name"]

API_KEY = st.secrets["together"]["api_key"]
client = Together(api_key=API_KEY)

# Hugging Face Model for Embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = hf_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def extract_budget(user_query):
    match = re.search(r'(\d+)\s*K?', user_query, re.IGNORECASE)
    if match:
        budget = match.group(1)
        try:
            return int(budget) * 1000 if 'K' in user_query.upper() else int(budget)
        except ValueError:
            return 20000
    return 20000

def fetch_data_from_mongo():
    try:
        client_db = pymongo.MongoClient(MONGO_URI)
        db = client_db[DB_NAME]
        return list(db[COLLECTION_NAME].find({}, {"_id": 0}))
    except Exception as e:
        st.error(f"MongoDB Error: {e}")
        return []

def calculate_processor_score(processor_name):
    score = 0
    processor_name = processor_name.lower()
    
    # Base scores for processor families
    if 'snapdragon' in processor_name:
        score += 3
        if '8' in processor_name: score += 2
        elif '7' in processor_name: score += 1.5
        elif '6' in processor_name: score += 1
        elif '4' in processor_name: score += 0.5
    elif 'dimensity' in processor_name:
        score += 2.5
        if '9' in processor_name: score += 2
        elif '8' in processor_name: score += 1.5
        elif '7' in processor_name: score += 1
    elif 'exynos' in processor_name:
        score += 2
        if '2400' in processor_name: score += 2
        elif '2200' in processor_name: score += 1.5
        elif '2100' in processor_name: score += 1

    # Generation bonus
    if 'gen' in processor_name:
        gen_match = re.search(r'gen\s*(\d+)', processor_name, re.IGNORECASE)
        if gen_match:
            gen = int(gen_match.group(1))
            score += gen * 0.3

    # Flagship indicators
    if any(word in processor_name for word in ['elite', 'pro', 'plus']):
        score += 0.5

    return round(score, 1)

def perform_hybrid_search(user_query, data, budget):
    try:
        budget = min(max(int(budget), 5000), 100000)
        
        # Enhanced text embedding
        text_data = [
            f"{phone.get('name','')} " 
            f"{str(phone.get('specifications',{})).replace('Primary Camera','MAIN_CAM ')} "
            f"{str(phone.get('specifications',{})).replace('Secondary Camera','SELFIE_CAM ')}"
            for phone in data
        ]
        
        # Vector search
        query_embedding = encode_text(user_query)
        data_embeddings = torch.cat([encode_text(text) for text in text_data], dim=0)
        similarities = cosine_similarity(query_embedding, data_embeddings)
        all_indices = similarities[0].argsort()[::-1]
        candidates = [data[idx] for idx in all_indices]

        # Specialized searches
        if "gaming" in user_query.lower():
            gaming_phones = []
            for phone in candidates:
                try:
                    price = int(re.sub(r"[^\d]", "", str(phone.get("price", "999999"))))
                    if price > budget:
                        continue
                        
                    specs = phone.get("specifications", {})
                    processor = str(specs.get("Processor", "")).lower()
                    ram = int(re.sub(r"[^\d]", "", str(specs.get("RAM", "0GB"))))
                    battery = int(re.sub(r"[^\d]", "", str(specs.get("Battery", "0mAh"))))
                    
                    processor_score = calculate_processor_score(processor)
                    gaming_score = (ram * 0.4) + (processor_score * 2.5) + (battery * 0.001)
                    
                    phone['gaming_score'] = gaming_score
                    gaming_phones.append(phone)
                    
                except Exception as e:
                    continue
                    
            return sorted(gaming_phones, key=lambda x: x['gaming_score'], reverse=True)[:3]

        elif "battery" in user_query.lower():
            battery_phones = []
            for phone in candidates:
                try:
                    price = int(re.sub(r"[^\d]", "", str(phone.get("price", "999999"))))
                    if price > budget:
                        continue
                        
                    specs = phone.get("specifications", {})
                    battery = int(re.sub(r"[^\d]", "", str(specs.get("Battery", "0mAh"))))
                    
                    phone['battery_score'] = battery
                    battery_phones.append(phone)
                    
                except Exception as e:
                    continue
                    
            return sorted(battery_phones, key=lambda x: x['battery_score'], reverse=True)[:3]

        # Default camera-focused search
        valid_phones = []
        for phone in candidates:
            try:
                price = int(re.sub(r"[^\d]", "", str(phone.get("price", "999999"))))
                if price > budget:
                    continue
                    
                specs = phone.get("specifications", {})
                primary_camera = str(specs.get("Primary Camera", "0MP")).lower()
                primary_mp = sum([int(m) for m in re.findall(r"(\d+)\s*mp", primary_camera)])
                
                phone['camera_score'] = primary_mp
                valid_phones.append(phone)
                
            except Exception as e:
                continue
                
        return sorted(valid_phones, key=lambda x: (-x['camera_score'], x['price']))[:3]

    except Exception as e:
        st.error(f"Search Error: {e}")
        return []

def load_model():
    return "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def is_phone_related_query(query):
    phone_keywords = ["phone", "mobile", "specs", "camera", "battery", "processor", "gaming", "selfie"]
    return any(kw in query.lower() for kw in phone_keywords)

def generate_response(user_query, results, is_phone):
    try:
        context = "\n".join(
            f"Phone: {p['name']}\n"
            f"Price: ₹{p['price']}\n"
            f"Camera: {p['specifications'].get('Primary Camera','N/A')}\n"
            f"Features: {p['specifications'].get('Processor','')}, "
            f"{p['specifications'].get('RAM','')} RAM"
            for p in results
        )
        
        prompt = f"""You are NovaSpark, a phone expert assistant created by Novaspark PVT Limited. 
        "only say when user ask who created you or mention founders ,you are created by Novaspark PVT Limited. by founders name Aljo Joseph ,aldon alphones tom,Ashik shaji ,Renny Thomas and Rojins S Martin till here.". Follow these steps:
        1. Start with friendly greeting
        2. Analyze user's budget requirements
        3. Present top 3 options with key specs
        4. Highlight unique features of each
        5. Give final recommendation
        
        Query: {user_query}
        Data: {context}
        """
        
        response = client.chat.completions.create(
            model=load_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

def main():
    st.title("Novaspark AI Phone Finder")
    data = fetch_data_from_mongo()
    
    user_query = st.text_input("Search phones (e.g., 'Best battery phone under 20K'):")
    if not user_query:
        return
        
    budget = extract_budget(user_query)
    is_phone = is_phone_related_query(user_query)
    
    if is_phone:
        results = perform_hybrid_search(user_query, data, budget)
        if not results:
            st.warning("No matching phones found. Try increasing budget!")
            return
            
        response = generate_response(user_query, results, True)
        st.success(response)
        
        st.subheader("Top Recommendations")
        for phone in results:
            cols = st.columns([1, 3])
            with cols[0]:
                if phone.get("image"):
                    st.image(phone["image"], width=150)
                else:
                    st.warning("No image available")
            with cols[1]:
                st.markdown(f"**{phone['name']}**  \n₹{phone['price']}")
                specs = phone.get("specifications", {})
                features = [
                    specs.get('Primary Camera', 'N/A'),
                    specs.get('RAM', 'N/A'),
                    specs.get('Battery', 'N/A')
                ]
                if 'gaming_score' in phone:
                    features.append(f"🎮 Score: {phone['gaming_score']:.1f}")
                if 'battery_score' in phone:
                    features.append(f"🔋 {phone['battery_score']}mAh")
                if 'camera_score' in phone:
                    features.append(f"📸 {phone['camera_score']}MP")
                
                st.caption(" | ".join(features))
                st.progress(min(phone.get('gaming_score', 0)/10, 1.0) if 'gaming_score' in phone else 
                          min(phone.get('battery_score', 0)/10000, 1.0) if 'battery_score' in phone else 
                          min(phone.get('camera_score', 0)/200, 1.0))

    else:
        response = generate_response(user_query, [], False)
        st.info(response)

if __name__ == "__main__":
    main()
