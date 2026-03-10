from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_groq import ChatGroq  # For using Groq's LLM
from secret_key import groq_api_key

# Import and Set it in the environment with the correct variable name
import os
os.environ["GROQ_API_KEY"] = groq_api_key


# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Updated model (mixtral-8x7b-32768 is deprecated)
    temperature=0,
    groq_api_key=groq_api_key
)

def generate_restaurant_name_and_items(cuisine):

    # Define prompts
    prompt_name = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that suggests creative restaurant names."),
        ("human", "Suggest one fancy name for a {cuisine} restaurant")
    ])

    prompt_items = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that suggests menu items."),
        ("human", "Suggest menu items for {restaurant_name}. Return as comma separated list")
    ])

    # Create chains
    name_chain = prompt_name | llm
    food_items_chain = prompt_items | llm

    # Execute sequentially (SequentialChain is deprecated, use this approach)
    # cuisine = "Arabic"

    # Step 1: Get restaurant name
    name_response = name_chain.invoke({"cuisine": cuisine})
    restaurant_name = name_response.content

    # Step 2: Get menu items
    items_response = food_items_chain.invoke({"restaurant_name": restaurant_name})
    menu_items = items_response.content

    # Output
    result = {
        "cuisine": cuisine,
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

    return result

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))

