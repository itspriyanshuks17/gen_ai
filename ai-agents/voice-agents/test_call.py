#!/usr/bin/env python3
import os
import jwt
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_token():
    """Generate VideoSDK JWT token"""
    api_key = os.getenv("VIDEOSDK_API_KEY")
    secret_key = os.getenv("VIDEOSDK_SECRET_KEY")
    
    payload = {
        "apikey": api_key,
        "permissions": ["allow_join", "allow_mod"],
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")

def test_call_agent():
    """Test calling the agent directly"""
    token = generate_token()
    
    url = "https://api.videosdk.live/v2/sip/call"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    
    # Test 1: Call agent directly
    data = {
        "to": "MyTelephonyAgent",
        "from": "+14155550123"
    }
    
    print(f"Testing call to agent...")
    print(f"Token: {token[:50]}...")
    print(f"Request: {data}")
    
    response = requests.post(url, headers=headers, json=data)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body: {response.text}")

if __name__ == "__main__":
    test_call_agent()
