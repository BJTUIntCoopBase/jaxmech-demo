"""Launch the jaxmech web frontend.

Usage:
    py -m jaxmech.web.run
    py -m jaxmech.web.run --port 9000
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser


def _open_browser(url: str, delay: float = 1.5) -> None:
    """Open *url* in the default browser after *delay* seconds."""
    time.sleep(delay)
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="jaxmech web frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not automatically open the browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        t = threading.Thread(target=_open_browser, args=(url,), daemon=True)
        t.start()

    import uvicorn
    uvicorn.run(
        "jaxmech.web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
