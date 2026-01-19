from google import genai
import os
import logging
import json
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(prompt: str, use_cache: bool = True) -> str:
    def get_limit_chars(provider_name: str) -> int:
        # Allow override
        env_val = os.getenv("LLM_MAX_PROMPT_CHARS")
        if env_val:
            try:
                return int(env_val)
            except:
                pass
        # Conservative defaults (chars ~ tokens*4)
        if provider_name == "openai":
            # 128k tokens -> ~512k chars; keep headroom
            return 350_000
        # Gemini 2.5 models can support very large contexts; keep a high cap
        return 1_200_000

    def get_chunk_size_chars(provider_name: str) -> int:
        env_val = os.getenv("LLM_CHUNK_SIZE_CHARS")
        if env_val:
            try:
                return int(env_val)
            except:
                pass
        # Keep per-request payloads well under limits
        if provider_name == "openai":
            return 120_000
        return 300_000

    def summarize_chunks_openai(text: str, client, model: str, max_chars: int) -> str:
        chunk_size = get_chunk_size_chars("openai")
        summaries = []
        instruction = (
            "You will compress a large, technical context for downstream analysis.\n"
            "Requirements:\n"
            "- Preserve key APIs, function/class names, file paths, and important code lines\n"
            "- Keep crucial semantics; remove boilerplate and repeated sections\n"
            "- Output concise plain text (no YAML fences)\n"
        )
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Compress this chunk:\n{chunk}"},
                ],
            )
            summaries.append(r.choices[0].message.content.strip())
        combined = "\n\n".join(summaries)
        # Iteratively reduce if still too big
        reduce_round = 0
        while len(combined) > max_chars and reduce_round < 3:
            reduce_round += 1
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Further compress while preserving technical fidelity:\n{combined}"},
                ],
            )
            combined = r.choices[0].message.content.strip()
        return combined

    def summarize_chunks_google(text: str, client, model: str, max_chars: int) -> str:
        chunk_size = get_chunk_size_chars("google")
        summaries = []
        instruction = (
            "You will compress a large, technical context for downstream analysis.\n"
            "Requirements:\n"
            "- Preserve key APIs, function/class names, file paths, and important code lines\n"
            "- Keep crucial semantics; remove boilerplate and repeated sections\n"
            "- Output concise plain text (no YAML fences)\n"
        )
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            resp = client.models.generate_content(
                model=model,
                contents=[instruction, f"Compress this chunk:\n{chunk}"],
            )
            summaries.append(resp.text.strip())
        combined = "\n\n".join(summaries)
        reduce_round = 0
        while len(combined) > max_chars and reduce_round < 3:
            reduce_round += 1
            resp = client.models.generate_content(
                model=model,
                contents=[instruction, f"Further compress while preserving technical fidelity:\n{combined}"],
            )
            combined = resp.text.strip()
        return combined

    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    # Select provider: "openai" or default to "google"
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "openai":
        # OpenAI
        # Env:
        #   - OPENAI_API_KEY
        #   - OPENAI_MODEL (default: gpt-4o-mini)
        from openai import OpenAI
        from openai import BadRequestError

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        max_chars = get_limit_chars("openai")
        effective_prompt = prompt
        if len(effective_prompt) > max_chars:
            logger.info(f"Prompt length {len(effective_prompt)} exceeds OpenAI limit ~{max_chars} chars; compressing...")
            effective_prompt = summarize_chunks_openai(effective_prompt, client, model, max_chars)
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": effective_prompt}],
            )
            response_text = r.choices[0].message.content
        except BadRequestError as e:
            # Fallback if context limit still exceeded due to tokenization overhead
            if "context length" in str(e).lower():
                logger.warning("Context length exceeded; applying additional compression and retrying once.")
                effective_prompt = summarize_chunks_openai(effective_prompt, client, model, max_chars // 2)
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": effective_prompt}],
                )
                response_text = r.choices[0].message.content
            else:
                raise
    else:
        # Google Gemini (AI Studio key)
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY", ""),
        )
        # model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        max_chars = get_limit_chars("google")
        effective_prompt = prompt
        if len(effective_prompt) > max_chars:
            logger.info(f"Prompt length {len(effective_prompt)} exceeds Gemini safe cap ~{max_chars} chars; compressing...")
            effective_prompt = summarize_chunks_google(effective_prompt, client, model, max_chars)
        response = client.models.generate_content(model=model, contents=[effective_prompt])
        response_text = response.text

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text


# # Use Azure OpenAI
# def call_llm(prompt, use_cache: bool = True):
#     from openai import AzureOpenAI

#     endpoint = "https://<azure openai name>.openai.azure.com/"
#     deployment = "<deployment name>"

#     subscription_key = "<azure openai key>"
#     api_version = "<api version>"

#     client = AzureOpenAI(
#         api_version=api_version,
#         azure_endpoint=endpoint,
#         api_key=subscription_key,
#     )

#     r = client.chat.completions.create(
#         model=deployment,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         max_completion_tokens=40000,
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

# # Use Anthropic Claude 3.7 Sonnet Extended Thinking
# def call_llm(prompt, use_cache: bool = True):
#     from anthropic import Anthropic
#     client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
#     response = client.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=21000,
#         thinking={
#             "type": "enabled",
#             "budget_tokens": 20000
#         },
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.content[1].text

# # Use OpenAI o1
# def call_llm(prompt, use_cache: bool = True):
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
#     r = client.chat.completions.create(
#         model="o1",
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

# Use OpenRouter API
# def call_llm(prompt: str, use_cache: bool = True) -> str:
#     import requests
#     # Log the prompt
#     logger.info(f"PROMPT: {prompt}")

#     # Check cache if enabled
#     if use_cache:
#         # Load cache from disk
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 logger.warning(f"Failed to load cache, starting with empty cache")

#         # Return from cache if exists
#         if prompt in cache:
#             logger.info(f"RESPONSE: {cache[prompt]}")
#             return cache[prompt]

#     # OpenRouter API configuration
#     api_key = os.getenv("OPENROUTER_API_KEY", "")
#     model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#     }

#     data = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}]
#     }

#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",
#         headers=headers,
#         json=data
#     )

#     if response.status_code != 200:
#         error_msg = f"OpenRouter API call failed with status {response.status_code}: {response.text}"
#         logger.error(error_msg)
#         raise Exception(error_msg)
#     try:
#         response_text = response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         error_msg = f"Failed to parse OpenRouter response: {e}; Response: {response.text}"
#         logger.error(error_msg)        
#         raise Exception(error_msg)
    

#     # Log the response
#     logger.info(f"RESPONSE: {response_text}")

#     # Update cache if enabled
#     if use_cache:
#         # Load cache again to avoid overwrites
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 pass

#         # Add to cache and save
#         cache[prompt] = response_text
#         try:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(cache, f)
#         except Exception as e:
#             logger.error(f"Failed to save cache: {e}")

#     return response_text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
