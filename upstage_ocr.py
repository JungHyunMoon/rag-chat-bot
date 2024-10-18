import os

import requests

def ocr_service(file):
    api_key = os.getenv("UPSTAGE_API_KEY")
    files = {"document": file}
    url = "https://api.upstage.ai/v1/document-ai/ocr"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, files=files)
    return response.json().get("text")


