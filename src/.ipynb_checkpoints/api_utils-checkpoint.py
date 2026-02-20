import requests
import time
import random
import math

from src.cost_utils import estimate_openai_cost

#api gpt model
def analyze_mult_images_label(base64_images: list, img_names: list, model_name: str, instruction_text: str, api_key: str):
    # Build content with numbered images
    content = [{"type": "text", "text": instruction_text}]
    for img_name, b64 in zip(img_names, base64_images):
        content.append({"type": "text", "text": img_name})  # label
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Create payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": content}
        ],
        "max_tokens": 600,
        "temperature": 0
    }
    
   # Extract response text
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()  # raise an error if request failed
    data = response.json()

    #extract usage
    usage = data.get("usage", {})
    usage_json = tokens_dict = {"prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "total_tokens": usage.get("total_tokens")}
    #calculate price
    cost = estimate_openai_cost(usage, model_name)
    
    return {"response": data["choices"][0]["message"]["content"],
            "usage": usage_json,
            "cost": cost}

#run model ordering image randomly and by batches
def analyze_batch(base64_images: list, img_names: list, model_name: str, instruction_text: str, api_key: str, batch_size: int, max_retries=5):
    # Combine and shuffle together to preserve alignment
    combined = list(zip(base64_images, img_names))
    random.shuffle(combined)
    base64_images, img_names = zip(*combined)  # unzip back
    base64_images, img_names = list(base64_images), list(img_names)
    
    # Calculate number of batches
    n_batches = math.ceil(len(base64_images) / batch_size)
    print(f"Total batches: {n_batches}")
    
    
    batches = {}
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_names = img_names[start:end]
        batches[f"batch{i+1}"] = batch_names
    
    results = {}
    for batch_label, batch_names in batches.items():
        indices = [img_names.index(name) for name in batch_names]
        batch_images = [base64_images[i] for i in indices]

        # Retry loop
        for attempt in range(max_retries):
            try:
                result = analyze_mult_images_label(
                                                    batch_images,
                                                    batch_names,
                                                    model_name,
                                                    instruction_text,
                                                    api_key)
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = (2 ** attempt) + random.random()
                    print(f"Rate limit hit. Waiting {wait:.1f}s before retry...")
                    time.sleep(wait)
                else:
                    raise
        else:
            #This else belongs to the "for attempt" loop — runs if never broken (i.e. all retries failed)
            raise Exception("Max retries exceeded for batch: " + batch_label)

        print("--")
        results[batch_label] = result  # return a list of
    
    return {"batches": batches,
            "results": results}

#to remove later
def analyze_batch_test(base64_images: list, img_names: list, model_name: str, instruction_text: str, api_key: str, batch_size: int):
    # Combine and shuffle together to preserve alignment
    combined = list(zip(base64_images, img_names))
    random.shuffle(combined)
    base64_images, img_names = zip(*combined)  # unzip back
    base64_images, img_names = list(base64_images), list(img_names)
    
    # Calculate number of batches
    n_batches = math.ceil(len(base64_images) / batch_size)
    print(f"Total batches: {n_batches}")
    
    batches = {}
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_names = img_names[start:end]
        batches[f"batch{i+1}"] = batch_names
    
    results = {}
    for batch_label, batch_names in batches.items():
        indices = [img_names.index(name) for name in batch_names]
        batch_images = [base64_images[i] for i in indices]
    
        print(batch_names)
    
        results[batch_label] = {"a"}  # return a list of
    
    return {"batches": batches,
            "results": results}
print("Functions loaded")