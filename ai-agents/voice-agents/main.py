import asyncio
import traceback
from videosdk.agents import Agent, AgentSession, RealTimePipeline, JobContext, RoomOptions, WorkerJob, Options
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from dotenv import load_dotenv
import os
import logging
import jwt
import time
logging.basicConfig(level=logging.INFO)

load_dotenv()

def generate_token():
    """Generate VideoSDK JWT token from API key and secret"""
    api_key = os.getenv("VIDEOSDK_API_KEY")
    secret_key = os.getenv("VIDEOSDK_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise ValueError("VIDEOSDK_API_KEY and VIDEOSDK_SECRET_KEY required in .env")
    
    payload = {
        "apikey": api_key,
        "permissions": ["allow_join", "allow_mod"],
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400  # 24 hours
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")

# Define the agent's behavior and personality
class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful AI assistant that answers phone calls. Keep your responses concise and friendly.",
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your real-time assistant. How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was great talking with you!")

async def start_session(context: JobContext):
    # Configure the Gemini model for real-time voice
    model = GeminiRealtime(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"]
        )
    )
    pipeline = RealTimePipeline(model=model)
    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)

    try:
        await context.connect()
        await session.start()
        await asyncio.Event().wait()
    finally:
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions()
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    try:
        # Generate token if not provided
        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not auth_token:
            auth_token = generate_token()
            logging.info("Generated VideoSDK auth token from API credentials")
        
        # Register the agent with a unique ID
        options = Options(
            auth_token=auth_token,
            agent_id="MyTelephonyAgent", # CRITICAL: Unique identifier for routing
            register=True, # REQUIRED: Register with VideoSDK for telephony
            max_processes=1, # Concurrent calls (use 1 for local/WSL)
            host="localhost",
            port=8081,
        )
        job = WorkerJob(entrypoint=start_session, jobctx=make_context, options=options)
        job.start()
    except Exception as e:
        traceback.print_exc()
