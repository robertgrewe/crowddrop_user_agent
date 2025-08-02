# Maps.py

import os
import requests
import json
import polyline
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any
from enum import Enum
import sys
import logging  # <--- Moved to the top to ensure it's always available

from langchain_core.tools import tool

from dotenv import load_dotenv, find_dotenv

# --- Corrected import for logging_config.py from a parent directory ---
# This ensures Python can find the logging_config module when this file is run directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables at the very beginning of the module
load_dotenv(find_dotenv())

# Now that the path is set, import the logging config
try:
    from logging_config import configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully imported logging_config from path: {project_root}")
except ModuleNotFoundError:
    # This block will run if the import still fails.
    # It will use a fallback configuration and log the failure.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"ModuleNotFoundError: Could not find 'logging_config'. Searched in: {sys.path}")
    logger.info("Using a basic fallback logging configuration.")


# Define an Enum for supported travel modes
class TravelMode(str, Enum):
    DRIVE = "DRIVE"
    WALK = "WALK"
    BICYCLE = "BICYCLE"
    TRANSIT = "TRANSIT"

# Define the input schema for the tool using Pydantic
class DirectionsInput(BaseModel):
    """Input schema for the get_Maps_directions_and_polyline_coords tool."""
    origin: str = Field(..., description="The starting point for the directions (e.g., 'Eiffel Tower, Paris'). Can be an address, plus code, or lat/long coordinates.")
    destination: str = Field(..., description="The ending point for the directions (e.g., 'Louvre Museum, Paris'). Can be an address, plus code, or lat/long coordinates.")
    travel_mode: TravelMode = Field(default=TravelMode.DRIVE, description="The mode of travel (e.g., 'WALK', 'DRIVE'). Defaults to 'DRIVE'.")

# Define the tool using the @tool decorator
# This decorator allows the function to be used as a tool in LangChain or similar frameworks.
# The corrected function from Maps.py

@tool(args_schema=DirectionsInput)
def get_Maps_directions_and_polyline_coords(origin: str, destination: str, travel_mode: str = TravelMode.DRIVE) -> str:
    """
    Retrieves directions between an origin and a destination using the
    Google Maps Platform Routes API (v2) and decodes the route's polyline
    into a list of latitude and longitude coordinates.

    Args:
        origin (str): The starting point for the directions.
        destination (str): The ending point for the directions.
        travel_mode (str): The mode of travel. Supported values are 'DRIVE', 'WALK', 'BICYCLE', 'TRANSIT'.

    Returns:
        str: A JSON string representing a dictionary with 'polyline_coordinates'
             (list of [latitude, longitude] pairs) and 'route_summary'
             (duration as string, distance_meters as int).
    """
    api_key = os.getenv("Maps_API_KEY")
    if not api_key:
        return json.dumps({"error": "Google Maps API key not found. Please set the Maps_API_KEY environment variable."})

    routes_api_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.duration,routes.distanceMeters"
    }

    # Safely get the string value for the API request
    api_travel_mode = travel_mode.value if isinstance(travel_mode, Enum) else travel_mode
    
    data = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": api_travel_mode,
        "computeAlternativeRoutes": False,
    }

    # Add routingPreference only for the DRIVE travel mode
    if api_travel_mode == "DRIVE":
        data["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"

    try:
        logger.info(f"Invoking get_Maps_directions_and_polyline_coords for origin: '{origin}', destination: '{destination}', mode: '{api_travel_mode}'")
        response = requests.post(routes_api_url, headers=headers, json=data)
        response.raise_for_status()
        routes_data = response.json()

        if not routes_data or 'routes' not in routes_data or not routes_data['routes']:
            logger.warning(f"No route found for '{origin}' to '{destination}' via {api_travel_mode}. Response: {routes_data}")
            return json.dumps({"error": f"No route found for '{origin}' to '{destination}' via {api_travel_mode}. Response: {routes_data}"})

        route = routes_data['routes'][0]
        encoded_polyline = route.get('polyline', {}).get('encodedPolyline')
        duration = route.get('duration')
        distance_meters = route.get('distanceMeters')

        if not encoded_polyline:
            logger.error("Encoded polyline not found in the API response.")
            return json.dumps({"error": "Encoded polyline not found in the API response."})

        decoded_coords = polyline.decode(encoded_polyline)

        output_data = {
            "polyline_coordinates": [[lat, lng] for lat, lng in decoded_coords],
            "route_summary": {
                "duration": duration,
                "distance_meters": distance_meters
            }
        }
        logger.info("Successfully retrieved and decoded route.")
        return json.dumps(output_data)

    except requests.exceptions.HTTPError as http_err:
        error_details = response.json() if response else {}
        error_message = error_details.get('error', {}).get('message', 'No specific message.')
        logger.error(f"HTTP error occurred: {http_err}. Details: {{API Error: {error_message}}}")
        return json.dumps({"error": f"HTTP error occurred: {http_err}. Details: {{API Error: {error_message}}}"})
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {e}"})
# --- Example Usage (for direct testing) ---
if __name__ == "__main__":
    logger.info("--- Testing Tool: Karlsruhe to Frankfurt (DRIVING) ---")
    drive_result = get_Maps_directions_and_polyline_coords.invoke(
        {"origin": "Karlsruhe, Germany", "destination": "Frankfurt, Germany", "travel_mode": "DRIVE"}
    )
    logger.info("Result for Driving:\n%s", json.dumps(json.loads(drive_result), indent=2))

    print("\n--- Testing Tool: Walking directions from Golden Gate Bridge to Pier 39, San Francisco ---")
    walk_result = get_Maps_directions_and_polyline_coords.invoke(
        {"origin": "Golden Gate Bridge, San Francisco", "destination": "Pier 39, San Francisco", "travel_mode": "WALK"}
    )
    print("Result for Walking:")
    print(json.dumps(json.loads(walk_result), indent=2))

    logger.info("\n--- Testing Tool: Walking directions from Nauener Tor to Brandenburger Str. ---")
    walk_result = get_Maps_directions_and_polyline_coords.invoke(
        {"origin": "Nauener Tor, Potsdam, Germany", "destination": "Brandenburger Str., 14467 Potsdam", "travel_mode": "WALK"}
    )
    logger.info("Result for Walking:\n%s", json.dumps(json.loads(walk_result), indent=2))

    logger.info("\n--- Testing Tool: Invalid Destination (still works as expected) ---")
    invalid_result = get_Maps_directions_and_polyline_coords.invoke(
        {"origin": "Karlsruhe, Germany", "destination": "NonExistentPlace123XYZ", "travel_mode": "DRIVE"}
    )
    logger.info("Result for Invalid Destination:\n%s", json.dumps(json.loads(invalid_result), indent=2))

    logger.info("\n--- Testing Tool: Missing API Key (simulate) ---")
    original_api_key = os.getenv("Maps_API_KEY")
    if original_api_key:
        del os.environ["Maps_API_KEY"]

    no_api_key_result = get_Maps_directions_and_polyline_coords.invoke(
        {"origin": "Karlsruhe, Germany", "destination": "Stuttgart, Germany", "travel_mode": "DRIVE"}
    )
    logger.info("Result for Missing API Key:\n%s", json.dumps(json.loads(no_api_key_result), indent=2))

    if original_api_key:
        os.environ["Maps_API_KEY"] = original_api_key