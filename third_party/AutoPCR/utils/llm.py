import json
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# PhenoRAG patch: httpx / dotenv / openai / groq / together were module-level imports upstream.
# Only ``openai`` is in this environment (environment.yaml), so importing this module would fail
# before any code ran -- including the local-model path, which needs no provider at all.
# The imports moved into the branch that uses them; see PATCHES.md.


def prompt(pr, model, api_provider='openai', api_key=None, seed=0, max_tokens=32):
    if not api_key:
        from dotenv import load_dotenv  # PhenoRAG patch: lazy, see PATCHES.md
        load_dotenv(os.path.dirname(__file__) + '/../.env')
        api_key = os.getenv(f'{api_provider.upper()}_API_KEY')
    if not hasattr(prompt, 'client') or prompt.api_key != api_key:
        if api_provider == 'openai':
            from openai import OpenAI
            prompt.client = OpenAI(api_key=api_key)
        elif api_provider == 'groq':
            from groq import Groq
            prompt.client = Groq(api_key=api_key)
        elif api_provider == 'together':
            from together import Together
            prompt.client = Together(api_key=api_key)
        elif api_provider == 'vllm':
            import httpx
            from openai import OpenAI
            prompt.client = OpenAI(api_key='EMPTY', base_url=api_key, http_client=httpx.Client(trust_env=False))
        prompt.api_key = api_key
    client = prompt.client

    if api_provider == 'vllm':
        vllm_concurrency = 16
        def one(item):
            i, p = item
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': p['system']},
                    {'role': 'user',   'content': p['user']},
                ],
                temperature=0,
                seed=seed,
                max_tokens=max_tokens,
            )
            return i, r.choices[0].message.content
        with ThreadPoolExecutor(max_workers=vllm_concurrency) as pool:
            return dict(pool.map(one, pr.items()))

    responses = {}
    for i in pr:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': pr[i]['system']},
                    {'role': 'user', 'content': pr[i]['user']}
                ],
                temperature=0,
                seed=seed,
                max_tokens=max_tokens
            )
        responses[i] = response.choices[0].message.content
    return responses