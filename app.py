"""Start the local web app and open it in the browser."""

import argparse
import logging
import threading
import time
import webbrowser

import httpx
import uvicorn

from api.main import create_app


def open_when_ready(url: str) -> None:
    """Open the jobs page, or the setup page while setup is unfinished."""
    for _ in range(60):
        try:
            ready = httpx.get(f"{url}/status", timeout=2.0).json()["ready"]
        except (httpx.HTTPError, ValueError, KeyError):
            time.sleep(0.5)
            continue
        webbrowser.open(url if ready else f"{url}/setup")
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Request logs would print full URLs, and some sources carry an API key in the URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    print(f"Job Search Agent is running at {url}. Press Ctrl+C to stop.")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
