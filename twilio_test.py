#!/usr/bin/env python3
"""
Comprehensive test script for Twilio integration.
"""
import os
import sys
import asyncio
import logging
import requests
from twilio.rest import Client
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telephony.config import (
    TWILIO_ACCOUNT_SID, 
    TWILIO_AUTH_TOKEN, 
    TWILIO_PHONE_NUMBER
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwilioTester:
    """Test utilities for Twilio integration."""
    
    def __init__(self):
        """Initialize tester."""
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    def test_credentials(self):
        """Test Twilio credentials."""
        try:
            # Test by fetching account info
            account = self.client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
            print(f"✓ Successfully connected to Twilio account: {account.friendly_name}")
            
            # Test phone number
            phone_numbers = self.client.incoming_phone_numbers.list(
                phone_number=TWILIO_PHONE_NUMBER
            )
            
            if phone_numbers:
                print(f"✓ Phone number {TWILIO_PHONE_NUMBER} is valid and active")
                return True
            else:
                print(f"✗ Phone number {TWILIO_PHONE_NUMBER} not found in your account")
                return False
                
        except Exception as e:
            print(f"✗ Error testing Twilio credentials: {e}")
            return False
    
    def test_local_server(self, base_url: str = 'http://localhost:5000'):
        """Test local server endpoints."""
        print(f"\nTesting local server at {base_url}...")
        
        # Test health endpoint
        try:
            response = requests.get(f'{base_url}/health')
            if response.status_code == 200:
                print(f"✓ Health check passed: {response.json()}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Cannot connect to server: {e}")
            return False
        
        # Test stats endpoint
        try:
            response = requests.get(f'{base_url}/stats')
            if response.status_code == 200:
                print(f"✓ Stats endpoint working: {response.json()}")
            else:
                print(f"✗ Stats endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Stats endpoint error: {e}")
        
        return True
    
    def make_test_call(self, to_number: str):
        """Make a test call."""
        try:
            call = self.client.calls.create(
                url='http://demo.twilio.com/docs/voice.xml',
                to=to_number,
                from_=TWILIO_PHONE_NUMBER
            )
            
            print(f"✓ Test call initiated. Call SID: {call.sid}")
            print(f"  Status: {call.status}")
            return call.sid
            
        except Exception as e:
            print(f"✗ Error making test call: {e}")
            return None
    
    def update_webhook_urls(self, webhook_base_url: str):
        """Update webhook URLs for the phone number."""
        try:
            # Find the phone number
            phone_numbers = self.client.incoming_phone_numbers.list(
                phone_number=TWILIO_PHONE_NUMBER
            )
            
            if not phone_numbers:
                print(f"✗ Phone number {TWILIO_PHONE_NUMBER} not found")
                return False
            
            phone_number = phone_numbers[0]
            
            # Update webhooks
            phone_number.update(
                voice_url=f'{webhook_base_url}/voice/incoming',
                voice_method='POST',
                status_callback=f'{webhook_base_url}/voice/status',
                status_callback_method='POST'
            )
            
            print(f"✓ Updated webhook URLs to {webhook_base_url}")
            print(f"  Voice URL: {webhook_base_url}/voice/incoming")
            print(f"  Status Callback: {webhook_base_url}/voice/status")
            return True
            
        except Exception as e:
            print(f"✗ Error updating webhook URLs: {e}")
            return False
    
    def check_ngrok(self):
        """Check if ngrok is running and get tunnel URL."""
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            tunnels = response.json()['tunnels']
            
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    print(f"✓ Ngrok tunnel found: {tunnel['public_url']}")
                    return tunnel['public_url']
            
            print("✗ No ngrok HTTPS tunnel found")
            return None
            
        except Exception:
            print("✗ Ngrok not running. Please start ngrok with: ngrok http 5000")
            return None

def main():
    """Run comprehensive tests."""
    print("=== Twilio Integration Test Suite ===\n")
    
    tester = TwilioTester()
    
    # Test credentials
    print("1. Testing Twilio credentials...")
    if not tester.test_credentials():
        print("\nPlease check your Twilio credentials in .env file")
        return
    
    # Test local server
    print("\n2. Testing local server...")
    if not tester.test_local_server():
        print("\nPlease start the server with: python twilio_app.py")
        return
    
    # Check ngrok
    print("\n3. Checking ngrok...")
    ngrok_url = tester.check_ngrok()
    
    if ngrok_url:
        # Update webhooks
        print("\n4. Updating webhook URLs...")
        tester.update_webhook_urls(ngrok_url)
    else:
        print("\nTo test with Twilio, start ngrok with: ngrok http 5000")
    
    # Optional test call
    print("\n5. Make a test call?")
    test_number = input("Enter a phone number to test (or press Enter to skip): ")
    
    if test_number:
        tester.make_test_call(test_number)
    
    print("\n=== Test Complete ===")
    print("\nNext steps:")
    print("1. Start the server: python twilio_app.py")
    print("2. Start ngrok: ngrok http 5000")
    print("3. Update webhooks with ngrok URL")
    print("4. Call your Twilio number to test")

if __name__ == '__main__':
    main()