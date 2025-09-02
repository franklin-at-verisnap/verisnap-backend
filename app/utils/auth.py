"""
Authentication utilities
"""
import os
import time
import jwt
import requests
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from config import settings


class AuthService:
    """Service for authentication operations"""
    
    def __init__(self):
        self.kid = settings.apple_kid
        self.team_id = settings.apple_team_id
        self.private_key_path = settings.private_key_path
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        if os.path.exists(settings.firebase_credentials_path):
            cred = credentials.Certificate(settings.firebase_credentials_path)
            try:
                firebase_admin.initialize_app(cred)
            except ValueError:
                # App already initialized
                pass
    
    def verify_firebase_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token"""
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            return decoded
        except Exception as e:
            print(f"Token verification failed: {e}")
            return None
    
    def create_apple_jwt(self) -> Optional[str]:
        """Create JWT for Apple DeviceCheck API"""
        if not all([self.kid, self.team_id, self.private_key_path]):
            print("Missing Apple JWT configuration")
            return None
        
        current_time = int(time.time())
        exp_time = current_time + 3600  # one hour later
        
        try:
            with open(self.private_key_path, 'r') as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read().encode(),
                    password=None,
                    backend=default_backend()
                )
            
            payload = {
                'iss': self.team_id,
                'iat': current_time,
                'exp': exp_time
            }
            
            token = jwt.encode(
                payload,
                private_key,
                algorithm='ES256',
                headers={'kid': self.kid}
            )
            
            return token
            
        except Exception as e:
            print(f"Error creating Apple JWT: {e}")
            return None
    
    def verify_device_token(self, device_token: str, transaction_id: str, timestamp: int) -> Dict[str, Any]:
        """Verify device token with Apple DeviceCheck API"""
        bearer = self.create_apple_jwt()
        if not bearer:
            return {'success': False, 'error': 'Failed to create Apple JWT'}
        
        url = "https://api.development.devicecheck.apple.com/v1/validate_device_token"
        headers = {
            'Authorization': f'Bearer {bearer}',
            'Content-Type': 'application/json'
        }
        payload = {
            'device_token': device_token,
            'transaction_id': transaction_id,
            'timestamp': timestamp
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return {'success': True}
            else:
                try:
                    error_data = response.json()
                    print(f"Error Verifying Device. Status: {response.status_code} Response: {error_data}")
                    return error_data
                except ValueError:
                    return {
                        'success': False,
                        'status_code': response.status_code,
                        'body': response.text,
                        'token_with_issue': device_token
                    }
        except Exception as e:
            print(f"Error verifying device token: {e}")
            return {'success': False, 'error': str(e)}
