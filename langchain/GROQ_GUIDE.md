# LangChain with Groq - Complete Guide

## Introduction to Groq

Groq is an AI inference platform that provides ultra-fast inference for large language models. When combined with LangChain, you can build powerful AI applications with exceptional speed and performance.

### Why Groq + LangChain?
- **Lightning Fast**: Sub-second response times
- **High Throughput**: Handle thousands of requests per second
- **Cost Effective**: Competitive pricing for high-volume applications
- **Easy Integration**: Simple API that works seamlessly with LangChain

## Installation & Setup

### 1. Install Required Packages

```bash
# Activate your virtual environment
source venv/bin/activate

# Install LangChain and Groq integration
pip install langchain langchain-groq python-dotenv
```

### 2. Get Your Groq API Key

1. Sign up at [Groq Console](https://console.groq.com/)
2. Generate an API key
3. Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Environment Setup

```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API key is loaded
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")
```

## Basic Examples

### 1. Simple Text Generation

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq chat model
# Available models: mixtral-8x7b-32768, llama2-70b-4096, gemma-7b-it
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Fast and capable model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=1024,  # Maximum response length
    api_key=os.getenv("GROQ_API_KEY")  # Your API key
)

# Generate a simple response
response = llm.invoke("Explain quantum computing in simple terms")

print("Response:", response.content)
print("Model used:", response.response_metadata.get('model_name'))
print("Tokens used:", response.response_metadata.get('token_usage'))
```

### 2. Streaming Responses

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize streaming-enabled model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    streaming=True,  # Enable streaming
    api_key=os.getenv("GROQ_API_KEY")
)

# Stream the response in real-time
print("Streaming response:")
for chunk in llm.stream("Write a short story about AI taking over the world"):
    print(chunk.content, end="", flush=True)
print("\n")  # New line after streaming
```

### 3. Chat Conversation with Memory

```python
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.8,  # Higher temperature for more creative responses
    api_key=os.getenv("GROQ_API_KEY")
)

# Create memory to store conversation history
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False  # Set to True to see chain execution details
)

# Have a multi-turn conversation
print("=== AI Tutor Conversation ===")

# First message
response1 = conversation.predict(input="Hi! I'm learning about machine learning. Can you explain what supervised learning is?")
print(f"Human: Hi! I'm learning about machine learning. Can you explain what supervised learning is?")
print(f"AI: {response1}")

# Follow-up question
response2 = conversation.predict(input="Can you give me a real-world example of supervised learning?")
print(f"\nHuman: Can you give me a real-world example of supervised learning?")
print(f"AI: {response2}")

# Another follow-up
response3 = conversation.predict(input="What's the difference between classification and regression?")
print(f"\nHuman: What's the difference between classification and regression?")
print(f"AI: {response3}")
```

## Chains with Groq

### 1. LLMChain for Structured Tasks

```python
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.3,  # Lower temperature for more focused responses
    api_key=os.getenv("GROQ_API_KEY")
)

# Create a prompt template for code explanation
code_prompt = PromptTemplate(
    input_variables=["language", "code_snippet"],
    template="""You are a senior {language} developer. Analyze this code snippet and explain:

1. What does this code do?
2. Key concepts demonstrated
3. Potential improvements

Code:
{code_snippet}

Provide a clear, educational explanation:"""
)

# Create the chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    verbose=True
)

# Example code to analyze
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

# Run the chain
result = code_chain.run({
    "language": "Python",
    "code_snippet": python_code
})

print("Code Analysis:")
print(result)
```

### 2. Sequential Chain: Content Creation Pipeline

```python
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model (using a different model for variety)
llm = ChatGroq(
    model_name="llama2-70b-4096",  # More capable model for complex tasks
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Chain 1: Generate blog post topic
topic_prompt = PromptTemplate(
    input_variables=["niche"],
    template="Generate an engaging blog post topic about {niche} that would attract readers. Make it specific and compelling."
)
topic_chain = LLMChain(
    llm=llm,
    prompt=topic_prompt,
    output_key="topic"
)

# Chain 2: Create outline
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Create a detailed outline for a blog post titled: {topic}

Include:
- Introduction hook
- 3-4 main sections with subsections
- Conclusion
- Call to action

Format as a numbered list."""
)
outline_chain = LLMChain(
    llm=llm,
    prompt=outline_prompt,
    output_key="outline"
)

# Chain 3: Write introduction
intro_prompt = PromptTemplate(
    input_variables=["topic", "outline"],
    template="""Based on this topic and outline, write an engaging 150-word introduction:

Topic: {topic}
Outline: {outline}

Make it compelling and encourage readers to continue reading."""
)
intro_chain = LLMChain(
    llm=llm,
    prompt=intro_prompt,
    output_key="introduction"
)

# Create sequential chain
blog_chain = SequentialChain(
    chains=[topic_chain, outline_chain, intro_chain],
    input_variables=["niche"],
    output_variables=["topic", "outline", "introduction"],
    verbose=True
)

# Run the content creation pipeline
result = blog_chain({
    "niche": "artificial intelligence in healthcare"
})

print("=== Blog Post Creation Pipeline ===")
print(f"Topic: {result['topic']}")
print(f"\nOutline:\n{result['outline']}")
print(f"\nIntroduction:\n{result['introduction']}")
```

## Agents with Groq

### 1. Simple Tool-Calling Agent

```python
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.1,  # Low temperature for reliable tool use
    api_key=os.getenv("GROQ_API_KEY")
)

# Define tools
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Search the web for current information. Use for questions about recent events, facts, or general knowledge."
    )
]

# Add memory for context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",  # Good for conversational agents
    memory=memory,
    verbose=True,
    handle_parsing_errors=True  # Handle occasional parsing issues gracefully
)

# Example queries
queries = [
    "What are the latest developments in quantum computing?",
    "Can you find information about the population of Tokyo?",
    "What's the current price of Bitcoin?"
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    try:
        response = agent.run(query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
```

### 2. Custom Tools with Groq

```python
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Custom tool: Get current weather
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap API"""
    # Note: You'll need to sign up for a free API key at openweathermap.org
    api_key = os.getenv("OPENWEATHER_API_KEY", "demo_key")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            humidity = data['main']['humidity']
            return f"Weather in {city}: {temp}°C, {description}, Humidity: {humidity}%"
        else:
            return f"Could not get weather for {city}. API response: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Custom tool: Calculate age
def calculate_age(birth_year: int) -> str:
    """Calculate age from birth year"""
    current_year = datetime.now().year
    age = current_year - birth_year
    return f"If born in {birth_year}, you would be {age} years old in {current_year}."

# Custom tool: Simple calculator
def calculate(expression: str) -> str:
    """Evaluate a simple mathematical expression"""
    try:
        # Only allow safe operations
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: Only numbers and basic operators (+, -, *, /) are allowed."

        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Define tools
tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather information for a city. Input should be a city name."
    ),
    Tool(
        name="AgeCalculator",
        func=calculate_age,
        description="Calculate age from birth year. Input should be a 4-digit year."
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Calculate mathematical expressions. Input should be a valid math expression."
    )
]

# Create agent with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
    max_iterations=3,  # Limit iterations to avoid long loops
    handle_parsing_errors=True
)

# Interactive example
print("=== Groq Agent with Custom Tools ===")
print("Try asking questions like:")
print("- What's the weather in London?")
print("- If I was born in 1990, how old am I?")
print("- What is 15 * 23 + 7?")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break

    try:
        response = agent.run(user_input)
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")
```

## Advanced Examples

### 1. RAG with Groq (Retrieval-Augmented Generation)

```python
from langchain_groq import ChatGroq
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.1,  # Low temperature for factual responses
    api_key=os.getenv("GROQ_API_KEY")
)

# Create sample documents (in a real scenario, load from files)
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "Groq provides ultra-fast inference for large language models with sub-second response times.",
    "Retrieval-Augmented Generation (RAG) combines retrieval from documents with generative AI.",
    "Vector databases store embeddings for efficient similarity search.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors."
]

# Create embeddings (using free HuggingFace model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vectorstore = FAISS.from_texts(documents, embeddings)

# Create custom prompt for RAG
rag_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Return top 3 similar documents
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True  # Return source documents for transparency
)

# Example queries
queries = [
    "What is LangChain?",
    "How fast is Groq?",
    "What is RAG and how does it work?",
    "What databases are mentioned?"
]

print("=== RAG with Groq Examples ===")
for query in queries:
    print(f"\nQuery: {query}")
    result = qa_chain({"query": query})

    print(f"Answer: {result['result']}")
    print("Sources:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  {i}. {doc.page_content[:100]}...")
```

### 2. Multi-Modal Content Generation

```python
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model_name="llama2-70b-4096",  # Using larger model for creative tasks
    temperature=0.8,  # Higher creativity
    api_key=os.getenv("GROQ_API_KEY")
)

# Chain 1: Generate story concept
concept_prompt = PromptTemplate(
    input_variables=["genre", "theme"],
    template="""Create a compelling story concept in the {genre} genre with the theme of {theme}.

Include:
- Main character description
- Central conflict
- Unique twist
- Setting details

Make it engaging and original."""
)
concept_chain = LLMChain(llm=llm, prompt=concept_prompt, output_key="concept")

# Chain 2: Write opening scene
scene_prompt = PromptTemplate(
    input_variables=["concept"],
    template="""Based on this story concept, write the opening scene (300-400 words):

{concept}

Focus on:
- Establishing the setting
- Introducing the main character
- Hinting at the central conflict
- Creating immediate engagement

Write in third-person limited perspective."""
)
scene_chain = LLMChain(llm=llm, prompt=scene_prompt, output_key="opening_scene")

# Chain 3: Generate character dialogue
dialogue_prompt = PromptTemplate(
    input_variables=["concept", "opening_scene"],
    template="""Review this story concept and opening scene:

CONCEPT: {concept}

OPENING SCENE: {opening_scene}

Now write a dialogue scene between the main character and a supporting character that:
- Reveals character personalities
- Advances the plot
- Includes subtext or foreshadowing
- Is natural and engaging

Write 8-10 lines of dialogue with action descriptions."""
)
dialogue_chain = LLMChain(llm=llm, prompt=dialogue_prompt, output_key="dialogue")

# Chain 4: Create marketing blurb
blurb_prompt = PromptTemplate(
    input_variables=["concept", "opening_scene"],
    template="""Create a compelling marketing blurb for this story:

CONCEPT: {concept}

OPENING SCENE: {opening_scene}

Write a 50-75 word blurb that would appear on a book cover or Amazon page.
Make it intriguing and genre-appropriate."""
)
blurb_chain = LLMChain(llm=llm, prompt=blurb_prompt, output_key="marketing_blurb")

# Create sequential chain
story_chain = SequentialChain(
    chains=[concept_chain, scene_chain, dialogue_chain, blurb_chain],
    input_variables=["genre", "theme"],
    output_variables=["concept", "opening_scene", "dialogue", "marketing_blurb"],
    verbose=True
)

# Generate a complete story package
result = story_chain({
    "genre": "cyberpunk",
    "theme": "artificial consciousness"
})

print("=== Complete Story Generation ===")
print(f"\nSTORY CONCEPT:\n{result['concept']}")
print(f"\nOPENING SCENE:\n{result['opening_scene']}")
print(f"\nDIALOGUE SCENE:\n{result['dialogue']}")
print(f"\nMARKETING BLURB:\n{result['marketing_blurb']}")
```

## Best Practices for Groq + LangChain

### 1. Model Selection
```python
# Choose the right model for your use case
models = {
    "mixtral-8x7b-32768": "Fast, general-purpose, good for most tasks",
    "llama2-70b-4096": "More capable, better for complex reasoning",
    "gemma-7b-it": "Efficient, good for instruction following"
}
```

### 2. Temperature Settings
```python
# Temperature guidelines
temperature_settings = {
    "creative_writing": 0.8-1.0,    # High creativity
    "code_generation": 0.1-0.3,     # Low randomness, high accuracy
    "chat_conversation": 0.5-0.7,   # Balanced creativity and coherence
    "factual_qa": 0.0-0.2          # Maximum accuracy
}
```

### 3. Error Handling
```python
from langchain_groq import ChatGroq
import time

def robust_groq_call(llm, prompt, max_retries=3):
    """Make a robust call to Groq with retry logic"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return None
```

### 4. Cost Optimization
```python
# Monitor token usage
def analyze_token_usage(response):
    """Analyze token usage from Groq response"""
    metadata = response.response_metadata
    prompt_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = metadata.get('token_usage', {}).get('total_tokens', 0)

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")

    # Groq pricing (as of 2024, subject to change)
    cost_per_token = 0.0000002  # $0.20 per million tokens
    estimated_cost = total_tokens * cost_per_token
    print(f"Estimated cost: ${estimated_cost:.6f}")
```

### 5. Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=30):
    """Decorator to rate limit function calls"""
    min_interval = 60 / calls_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)  # Groq's rate limit
def safe_groq_call(llm, prompt):
    return llm.invoke(prompt)
```

## Performance Comparison

| Feature | Groq + LangChain | Traditional APIs |
|---------|------------------|------------------|
| Response Time | Sub-second | 5-30 seconds |
| Throughput | 1000+ req/min | 10-100 req/min |
| Cost | Low | Medium-High |
| Setup Complexity | Low | Medium |
| Streaming | Yes | Limited |

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```python
   # Ensure .env file exists and is loaded
   from dotenv import load_dotenv
   import os
   load_dotenv()
   assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not found"
   ```

2. **Rate Limiting**
   ```python
   # Implement exponential backoff
   import time
   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               if i == max_retries - 1:
                   raise e
               time.sleep(2 ** i)
   ```

3. **Model Not Available**
   ```python
   # Check available models
   available_models = ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"]
   chosen_model = "mixtral-8x7b-32768"  # Default fallback
   ```

## Next Steps

1. **Experiment**: Try different models and temperature settings
2. **Scale**: Implement proper error handling and rate limiting
3. **Monitor**: Track token usage and response times
4. **Optimize**: Choose appropriate models for different tasks
5. **Deploy**: Build production applications with Groq + LangChain

## Resources

- [Groq Documentation](https://console.groq.com/docs)
- [LangChain Groq Integration](https://python.langchain.com/docs/integrations/llms/groq)
- [Groq API Reference](https://console.groq.com/docs/api)
- [LangChain Documentation](https://python.langchain.com/)

Remember: Groq's speed makes it perfect for real-time applications, chatbots, and high-throughput scenarios. Combine it with LangChain's flexibility to build powerful AI applications!</content>
<parameter name="filePath">/home/virtualuser/gen_ai/langchain/GROQ_GUIDE.md