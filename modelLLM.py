import streamlit as st
import pymongo
import json
from together import Together
import re  # For extracting phone names from LLM response

# Configuration
MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db_name"]
COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
API_KEY = st.secrets["together"]["api_key"]
client = Together(api_key=API_KEY)

def fetch_all_phones():
    """Fetch all phone data from MongoDB."""
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return list(collection.find({}, {"_id": 0}))

# Initialize clients
mongo_client = pymongo.MongoClient(MONGO_URI)
llm_client = Together(api_key=API_KEY)
def llm_search(query: str, phones: list):
    """Let the LLM analyze the query and select the best phones from the database."""
    
    # Format phone data concisely (remove images from LLM input)
    phone_data = "\n".join(
        f"{idx+1}. {p['name']} (₹{p['price']}) | {p['specifications'].get('Processor', 'N/A')}, "
        f"{p['specifications'].get('RAM', 'N/A')} RAM, {p['specifications'].get('Battery', 'N/A')} battery, "
        f"{p['specifications'].get('Primary Camera', 'N/A')} Camera"
        for idx, p in enumerate(phones)
    )

    prompt = f"""You are NovaSpark, a phone expert. 
    A user has asked: {query}

    Here is a list of available phones:
    {phone_data}

    Your task is to analyze the user's query and recommend the best phones based on their requirements. Follow these guidelines:

    1. **Understand the User's Needs:**
    - Identify key aspects of the query, such as budget, brand preference, use case (e.g., gaming, camera, battery life), or specific features (e.g., processor, RAM, storage).
    - If the user mentions a budget (e.g., "under ₹20,000"), prioritize phones within that range.
    - If the user mentions a brand, prioritize phones from that brand. If the brand is unavailable, suggest alternatives and clearly state that they are not the requested brand.

    2. **Prioritize Based on Use Case:**
    - **Gaming Phones:** Prioritize phones with powerful processors (e.g., Snapdragon 8 series, Dimensity high-end chips) and good cooling systems. Look for high refresh rate displays and ample RAM.
    - **Camera Phones:** Prioritize phones with high-resolution cameras, multiple lenses, and advanced camera features (e.g., optical zoom, night mode).
    - **Battery Life:** Prioritize phones with large battery capacities and efficient processors.
    - **All-Rounders:** Recommend phones that balance performance, camera quality, battery life, and price.

    3. **Price-to-Performance Ratio:**
    - For budget-conscious users, recommend phones that offer the best value for money. Highlight phones with good specifications at lower prices.
    - For users with higher budgets, prioritize premium features and performance.

    4. **Comparison and Recommendations:**
    - Select the top 3 phones that best match the user's requirements.
    - Provide a comparison of the selected phones, highlighting their key features, pros, and cons.
    - If the user's requested brand or specific feature is unavailable, suggest alternatives and explain why they are good options.

    5. **Include Images:**
    - Ensure that the image of each recommended phone is included in the response.

    6. **Keep It Concise:**
    - Limit your response to 150 words or less. Focus on the most important details.

    Final Output:
    - Provide a clear and concise response with the top 3 recommendations.
    - Include a comparison table or bullet points highlighting the key features of each phone.
    - Ensure that the response is well-structured and in table format.
        -explan like a phone expert.show Selected Phones at last along with response.
- List selected phones EXACTLY as they appear in the database
- Use this exact format for final selection:
    **Selected Phones:**
    - Exact Phone Name 1
    - Exact Phone Name 2
    - Exact Phone Name 3
"""
    """

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        llm_output = response.choices[0].message.content

        # Extract selected phone names from LLM response
        selected_phones = []
        match = re.search(r"\*\*Selected Phones:\*\*(.*?)$", llm_output, re.S)
        if match:
            selected_phones = [p.strip("- ").strip() for p in match.group(1).split("\n") if p.strip()]

        # Match selected phones with full dataset (including images)
        matched_phones = [p for p in phones if p["name"] in selected_phones]

        return llm_output, matched_phones
    
    except Exception as e:
        return f"Error generating response: {str(e)}", []
def main():
    st.title("NovaSpark AI Phone Expert")
    
    if "phones" not in st.session_state:
        st.session_state.phones = fetch_all_phones()
    
    query = st.text_input("Describe your ideal phone:", 
                          placeholder="e.g. 'Best gaming phone under ₹20,000'")

    if query:
        with st.spinner("Searching for the best phones..."):
            response_text, results = llm_search(query, st.session_state.phones)
        
        st.success(response_text)

        if results:
            st.subheader("Top Recommended Phones")
            for phone in results:
                with st.expander(f"{phone['name']} - ₹{phone['price']}"):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if phone.get("image"):
                            st.image(phone["image"], width=150)
                        else:
                            st.warning("No image available")
                    with cols[1]:
                        specs = phone["specifications"]
                        st.markdown(f"""
                        **Key Specifications:**
                        - **Camera**: {specs.get('Primary Camera', 'N/A')}
                        - **Processor**: {specs.get('Processor', 'N/A')}
                        - **Battery**: {specs.get('Battery', 'N/A')}
                        - **RAM/Storage**: {specs.get('RAM', 'N/A')}/{specs.get('Storage', 'N/A')}
                        """)

if __name__ == "__main__":
    main()
