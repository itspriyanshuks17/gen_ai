import streamlit as st
import langchain_helper

# Page config
st.set_page_config(page_title="🍽️ Restaurant Generator", page_icon="🍽️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 3.5rem;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2rem;
}
.restaurant-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.menu-item {
    background: black;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border-left: 4px solid #FF6B6B;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🍽️ AI Restaurant Generator</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🌍 Select Cuisine")
cuisine = st.sidebar.selectbox(
    "Choose your style:",
    ("Indian 🇮🇳", "Italian 🇮🇹", "Mexican 🇲🇽", "Arabic 🇸🇦", "American 🇺🇸")
)

if cuisine:
    cuisine_clean = cuisine.split()[0]
    
    with st.spinner('✨ Creating magic...'):
        response = langchain_helper.generate_restaurant_name_and_items(cuisine_clean)
    
    # Restaurant name card
    st.markdown(f'''
    <div class="restaurant-card">
        <h2>🏪 {response["restaurant_name"].strip()}</h2>
        <p>Authentic {cuisine_clean} Cuisine</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Menu section
    st.markdown("### 🍴 Signature Menu")
    menu_items = [item.strip() for item in response['menu_items'].strip().split(",")]
    
    for item in menu_items:
        st.markdown(f'<div class="menu-item">🍽️ {item}</div>', unsafe_allow_html=True)
else:
    st.info("👈 Select a cuisine to start generating!")

