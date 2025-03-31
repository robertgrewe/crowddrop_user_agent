import requests
import urllib.parse
import json
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # Load environment variables from .env

class CrowdDropServices:
    def __init__(self, api_base_url=None, username=None, password=None):
        """
        Initializes the CrowdDropServices class.

        Args:
            api_base_url (str, optional): The base URL of the Crowddrop API. If None, it will be read from the .env file.
            username (str, optional): The username for authentication. If None, it will be read from the .env file.
            password (str, optional): The password for authentication. If None, it will be read from the .env file.
        """
        if api_base_url is None:
            self.api_base_url = os.getenv("API_BASE_URL")
            if self.api_base_url is None:
                print("API_BASE_URL not found in .env file or as a provided argument. Using default.")
                self.api_base_url = "https://dev.crowddrop.aidobotics.ai/app"  # Default value.
        else:
            self.api_base_url = api_base_url

        if username is None:
            self.username = os.getenv("CROWDDROP_USERNAME")
            if self.username is None:
                raise ValueError("CROWDDROP_USERNAME not found in .env file or as a provided argument.")
        else:
            self.username = username

        if password is None:
            self.password = os.getenv("CROWDDROP_PASSWORD")
            if self.password is None:
                raise ValueError("CROWDDROP_PASSWORD not found in .env file or as a provided argument.")
        else:
            self.password = password

    def authenticate_crowddrop(self):
        """
        Authenticates with the Crowddrop API and returns the authentication token.

        Returns:
            str: The authentication token (id_token) if successful, None otherwise.
        """
        auth_url = f"{self.api_base_url}/auth/login"

        # URL-encode the username and password
        encoded_username = urllib.parse.quote(self.username)
        encoded_password = urllib.parse.quote(self.password)

        full_url = f"{auth_url}?username={encoded_username}&password={encoded_password}"

        headers = {
            'accept': 'application/json',
        }

        try:
            response = requests.post(full_url, headers=headers, data='')
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            response_json = response.json()
            return response_json.get('id_token')

        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to decode json: {e}")
            print(f"response text: {response.text}")
            return None
        except KeyError as e:
            print(f"Key error: {e}")
            print(f"response text: {response.text}")
            return None

if __name__ == "__main__":
    """
    Example usage of the CrowdDropServices class.
    """

    crowddrop_service = CrowdDropServices()  # Initializes the class. Username and password are now pulled from the .env file.
    token = crowddrop_service.authenticate_crowddrop()

    if token:
        print(f"Authentication successful. Token: {token}")
        # You can now use the 'token' variable for subsequent API requests
        # Example:
        # headers = {'Authorization': f'Bearer {token}', 'accept': 'application/json'}
        # ... make other API calls using crowddrop_service and the token ...
    else:
        print("Authentication failed.")