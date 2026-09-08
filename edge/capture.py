import os
import subprocess
import time

import requests

# --- Configuration ---
CAMERA_ID = os.environ.get("CAMERA_ID", "mess_main")
SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.1.100:8000")
API_KEY = os.environ.get("CAMERA_API_KEY", "change-me")
INTERVAL = 10 # seconds between captures
IMAGE_WIDTH = 736
IMAGE_HEIGHT = 490

API_ENDPOINT = f"{SERVER_URL}/ingest/{CAMERA_ID}"

def capture_and_send():
    """Captures a frame using libcamera and sends it to the central server."""
    try:
        # We use libcamera-still to capture a JPEG. 
        # -n: no preview (headless)
        # --width, --height: resize to match server expectation
        # --immediate: capture as quickly as possible
        # -o -: output to stdout (avoids wearing out the SD card with constant writes)
        cmd = [
            "libcamera-still",
            "-n",
            "--width", str(IMAGE_WIDTH),
            "--height", str(IMAGE_HEIGHT),
            "--immediate",
            "-o", "-"
        ]
        
        # Capture the image data into memory
        result = subprocess.run(cmd, capture_output=True, check=True)
        image_data = result.stdout
        
        if not image_data:
            print("Warning: libcamera returned no data.")
            return

        # Prepare HTTP request
        files = {
            'file': ('capture.jpg', image_data, 'image/jpeg')
        }
        headers = {
            'X-Camera-Key': API_KEY
        }
        
        # Send to API
        response = requests.post(API_ENDPOINT, files=files, headers=headers, timeout=10)
        response.raise_for_status()
        
        print(f"[{time.strftime('%H:%M:%S')}] Sent frame. Status: {response.status_code}")
        
    except subprocess.CalledProcessError as e:
        print(f"Camera Error: {e.stderr.decode('utf-8', errors='ignore')}")
    except requests.exceptions.RequestException as e:
        print(f"Network Error: Could not connect to {API_ENDPOINT} ({e})")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def main():
    print("Starting capture loop...")
    print(f"Camera ID: {CAMERA_ID}")
    print(f"Endpoint: {API_ENDPOINT}")
    
    while True:
        start_time = time.time()
        
        capture_and_send()
        
        # Ensure we wait exactly INTERVAL seconds between iterations
        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
