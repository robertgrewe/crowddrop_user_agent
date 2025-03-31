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

        self.token = None  # Added token as a property

    def authenticate(self):
        """
        Authenticates with the Crowddrop API and returns the authentication token.
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
            self.token = response_json.get('id_token')  # Assign token to self.token
            return self.token

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

    def get_tasks(self, page=1, size=10):
        """
        Retrieves tasks from the Crowddrop API with pagination and authorization.
        """
        if self.token is None:
            print("Authentication token is missing. Please authenticate first.")
            return None

        url = f"{self.api_base_url}/tasks/"

        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.token}',
        }

        params = {
            'page': page,
            'size': size,
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to decode json: {e}")
            print(f"response text: {response.text}")
            return None

    def get_task(self, task_id):
        """
        Retrieves a single task from the Crowddrop API by its ID.
        """
        if self.token is None:
            print("Authentication token is missing. Please authenticate first.")
            return None

        url = f"{self.api_base_url}/tasks/{task_id}"

        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.token}',
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to decode json: {e}")
            print(f"response text: {response.text}")
            return None

if __name__ == "__main__":
    """
    Example usage of the CrowdDropServices class.
    """

    crowddrop_service = CrowdDropServices()  # Initializes the class. Username and password are now pulled from the .env file.

    if crowddrop_service.authenticate():
        print(f"Authentication successful. Token: {crowddrop_service.token}")

        tasks = crowddrop_service.get_tasks(page=1, size=10)  # Call get_tasks here.

        if tasks:
            print("All tasks:")
            print(json.dumps(tasks, indent=2))
        else:
            print("Failed to retrieve tasks.")

        task_id = "67b8760e920af4b7a5ba837f"  # Replace with a valid task ID
        task = crowddrop_service.get_task(task_id)

        if task:
            print(f"\nTask with ID {task_id}:")
            print(json.dumps(task, indent=2))
        else:
            print(f"Failed to retrieve task with ID {task_id}.")

    else:
        print("Authentication failed.")