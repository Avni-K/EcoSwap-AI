import sys
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from safetensors.torch import load_file

MODEL_DIR = "saved_model"
CLASSES_FILE = "classes.json"

def search_ddg_manual(query, current_links):
    """
    Helper function to scrape DuckDuckGo Lite.
    Returns a list of NEW Amazon links found.
    """
    base_url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://duckduckgo.com/"
    }
    
    new_links = []
    print(f"  > Searching for: '{query}'...")

    try:
        response = requests.post(base_url, data={'q': query}, headers=headers, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__a')
        if not results:
            results = soup.find_all('a', href=True)

        for link_tag in results:
            raw_url = link_tag.get('href')
            if not raw_url: continue

            # --- CLEANING LOGIC ---
            clean_url = raw_url

            # Skip Ads
            if "/y.js" in raw_url or "ad_provider" in raw_url:
                continue

            # Unwrap Redirects
            if "uddg=" in raw_url:
                try:
                    clean_url = unquote(raw_url.split("uddg=")[1].split("&")[0])
                except IndexError:
                    continue

            # Validation: Must be Amazon and Unique
            if "amazon.com" in clean_url and clean_url.startswith("http"):
                # Simplify URL (remove tracking refs)
                if "/ref=" in clean_url:
                    clean_url = clean_url.split("/ref=")[0]
                
                # Check uniqueness against both current batch and global list
                if clean_url not in current_links and clean_url not in new_links:
                    new_links.append(clean_url)
            
            # Stop if we found enough for this batch (we need 3 total)
            if len(current_links) + len(new_links) >= 3:
                break
                
    except Exception:
        return []

    return new_links

def get_eco_alternatives_robust(product_name):
    """
    Tries multiple search queries to ensure we get 3 links.
    """
    # 1. Clean the name (remove underscores)
    clean_name = product_name.replace("_", " ")
    
    # 2. Define strategies: Specific first, then broad
    queries = [
        f"eco friendly alternatives to {clean_name} amazon",
        f"compostable {clean_name} amazon",
        f"sustainable {clean_name} amazon" 
    ]
    
    all_links = []
    
    print(f"\nSearching for eco-friendly alternatives for '{clean_name}'...")
    
    # 3. Loop through queries until we have 3 links
    for q in queries:
        if len(all_links) >= 3:
            break
            
        found_links = search_ddg_manual(q, all_links)
        all_links.extend(found_links)
        
    return all_links[:3] # Return top 3

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_single.py <image_path>")
        exit(1)

    image_path = sys.argv[1]
    device = torch.device("cpu") 

    # --- Load Model & Processor ---
    try:
        config = ViTConfig.from_pretrained(MODEL_DIR)
        model = ViTForImageClassification(config)
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(model.config.hidden_size, config.num_labels)
        )
        try:
            state_dict = load_file(f"{MODEL_DIR}/model.safetensors")
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            try:
                state_dict = torch.load(f"{MODEL_DIR}/pytorch_model.bin", map_location=device)
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                print("Error: Model weights not found.")
                exit(1)
        model.to(device)
        model.eval()
        
        processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
        with open(CLASSES_FILE, "r") as f:
            idx_to_class = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"Setup Error: {e}")
        exit(1)

    # --- Predict ---
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = idx_to_class[pred_idx]
            print(f"Predicted: {pred_class} ({probs[pred_idx].item():.2f})")
    except Exception as e:
        print(f"Prediction Error: {e}")
        exit(1)

    # --- Find Alternatives ---
    if pred_class:
        alternatives = get_eco_alternatives_robust(pred_class)
        
        if alternatives:
            print(f"\nSUCCESS: Found {len(alternatives)} Amazon alternatives:")
            for i, link in enumerate(alternatives, 1):
                print(f"{i}. {link}")
        else:
            print("\nNo Amazon links found after trying multiple queries.")

if __name__ == "__main__":
    main()