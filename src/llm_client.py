import os
import httpx
from typing import Optional
from src.config import config

def generate(prompt: str, json_mode: bool = False, system_prompt: str = None) -> str:
    """
    Generates a response from the configured LLM API.
    Supports Nvidia NIM, OpenRouter, Groq, and local provider.
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    # base_url = os.environ.get("BASE_URL")
    
    # Precedence: Nvidia NIM > OpenRouter > Groq > Local
    if nvidia_key:
        api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key = nvidia_key
        model = "openai/gpt-oss-120b"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    elif openrouter_key:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = openrouter_key
        model = "meta-llama/llama-3-8b-instruct"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "AliveGraphRAG"
        }
    elif groq_key:
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        api_key = groq_key
        model = config.llm.model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    # elif base_url:
    #     resolved_base = base_url.strip()
    #     if "0.0.0.0" in resolved_base:
    #         resolved_base = resolved_base.replace("0.0.0.0", "127.0.0.1")
    #     if resolved_base.endswith("/"):
    #         resolved_base = resolved_base[:-1]
        
    #     api_url = f"{resolved_base}/v1/chat/completions"
    #     api_key = "local"
    #     model = config.llm.model
    #     headers = {
    #         "Authorization": f"Bearer {api_key}",
    #         "Content-Type": "application/json"
    #     }
    else:
        raise ValueError("No LLM provider configuration found in environment variables.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif json_mode:
        messages.append({
            "role": "system", 
            "content": "You are a helpful assistant that outputs only valid JSON schemas as requested."
        })
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }
    
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=60.0) as client:
        response = client.post(api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"LLM API returned status {response.status_code}: {response.text}")
            
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

def health_check() -> bool:
    """
    Verifies that the LLM client is configured and can communicate with the API.
    """
    try:
        res = generate("ping")
        return len(res) > 0
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
