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

def make_outbound_call(gateway_id, phone_number):
    """Make an outbound call from agent to phone number"""
    token = generate_token()
    
    url = "https://api.videosdk.live/v2/sip/call"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    
    data = {
        "gatewayId": gateway_id,
        "sipCallTo": phone_number
    }
    
    print(f"Making outbound call...")
    print(f"Gateway ID: {gateway_id}")
    print(f"Calling: {phone_number}")
    print(f"Agent: MyTelephonyAgent")
    
    response = requests.post(url, headers=headers, json=data)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("\n✓ Call initiated successfully!")
        print("Your phone should ring shortly...")
    else:
        print("\n✗ Call failed!")

if __name__ == "__main__":
    # Default values - you can change these
    GATEWAY_ID = input("Enter your Outbound Gateway ID (from VideoSDK dashboard): ").strip()
    PHONE_NUMBER = input("Enter phone number to call [default: +16303496813]: ").strip() or "+16303496813"
    
    make_outbound_call(GATEWAY_ID, PHONE_NUMBER)
