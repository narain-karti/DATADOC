import subprocess
import time
import requests
import sys

def main():
    print("Starting DATADOC UI server in the background...")
    # Start the ui server
    process = subprocess.Popen(
        ["uv", "run", "datadoc", "ui", "scratch/dirty_data.csv", "--port", "8010"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to boot up
    time.sleep(5)
    
    headers = {"local": "X-DATADOC-SESSION"}
    base_url = "http://127.0.0.1:8010/api"
    
    try:
        print("Testing /dataset/metadata...")
        res = requests.get(f"{base_url}/dataset/metadata", headers=headers)
        res.raise_for_status()
        print("Metadata:", res.json()["columns"], "columns")
        
        print("Testing /pipeline/profile...")
        res = requests.get(f"{base_url}/pipeline/profile?target=churn", headers=headers)
        res.raise_for_status()
        print("Profile successful.")
        
        print("Testing /pipeline/plan...")
        res = requests.post(
            f"{base_url}/pipeline/plan", 
            json={"target": "churn", "task": "auto", "drop_identifiers": True, "scaling": "standard", "clip_outliers": True},
            headers=headers
        )
        res.raise_for_status()
        print("Plan operations:", len(res.json()["operations"]))
        
        print("Testing /pipeline/fit...")
        res = requests.post(
            f"{base_url}/pipeline/fit", 
            json={"target": "churn", "task": "auto", "drop_identifiers": True, "scaling": "standard", "clip_outliers": True},
            headers=headers
        )
        res.raise_for_status()
        print("Fit successful. Output columns:", res.json()["columns"])
        
        print("Testing /pipeline/preview...")
        res = requests.get(f"{base_url}/pipeline/preview", headers=headers)
        res.raise_for_status()
        print("Preview successful.")
        
        print("All API tests passed!")
        
    except Exception as e:
        print("API test failed:", e)
        if hasattr(e, "response") and e.response is not None:
            print("Response:", e.response.text)
        sys.exit(1)
    finally:
        print("Shutting down the server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
