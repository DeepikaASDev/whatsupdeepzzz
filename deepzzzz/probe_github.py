"""Diagnose GitHub Models connection problems:  python -m deepzzzz.probe_github

Sends one tiny request to each GitHub Models endpoint and prints what comes back,
then tells you which .env settings to use.
"""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

ENDPOINTS = [
    # (base URL, model name to send)
    ("https://models.github.ai/inference", "openai/gpt-4o-mini"),
    ("https://models.inference.ai.azure.com", "gpt-4o-mini"),
]


def main() -> int:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN is empty in .env")
        return 1
    print(f"Token: {token[:11]}...{token[-4:]}")
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
        if os.getenv(var):
            print(f"Note: {var} is set to {os.getenv(var)} (requests go through a proxy)")

    working = None
    for base, model in ENDPOINTS:
        url = base + "/chat/completions"
        print(f"\n== {url}  (model {model})")
        try:
            r = httpx.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                         "Accept": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "Say hi in 3 words."}],
                      "max_tokens": 20},
                timeout=30,
                follow_redirects=False,
            )
        except Exception as exc:
            print(f"   request failed: {type(exc).__name__}: {exc}")
            continue
        print(f"   status: {r.status_code}")
        print(f"   content-type: {r.headers.get('content-type')}")
        if r.is_redirect:
            print(f"   redirected to: {r.headers.get('location')}")
        print(f"   body: {r.text[:300]!r}")
        if r.status_code == 200 and "json" in (r.headers.get("content-type") or ""):
            try:
                reply = r.json()["choices"][0]["message"]["content"]
                print(f"   WORKS. Model said: {reply!r}")
                working = working or (base, model)
            except Exception:
                pass

    print()
    if working:
        base, model = working
        print("Use these lines in .env:")
        print(f"   LLM_MODEL=github:{model}")
        if base != ENDPOINTS[0][0]:
            print(f"   GITHUB_MODELS_ENDPOINT={base}")
        return 0
    print("Neither endpoint worked. Send a screenshot of this output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
