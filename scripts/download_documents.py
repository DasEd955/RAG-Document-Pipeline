"""download_documents.py - CLI for downloading documents from URLs and saving them as HTML.

Extracts URLs from a markdown file, fetches them using requests (or Playwright
for JS-heavy pages), and saves the HTML to a directory with metadata tracking.
"""
import argparse
import re
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import requests
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


def extract_urls_from_markdown(md_path: Path) -> list:
    """Extract and deduplicate URLs from a markdown file.

    Uses a regex to find all http(s) URLs, strips trailing punctuation (.,;|),
    and returns a deduplicated list in order.

    Args:
        md_path (Path): Path to a markdown file (e.g., planning.md).

    Returns:
        list: A list of unique URL strings found in the markdown.

    Example:
        >>> urls = extract_urls_from_markdown(Path("planning.md"))
        >>> print(urls[0])
        https://www.reddit.com/r/PennStateUniversity/...
    """
    text = md_path.read_text(encoding="utf-8")
    # Crude URL regex that skips trailing table delimiters
    found = re.findall(r"https?://[^\s)\]\|]+", text)
    cleaned = []
    seen = set()
    for u in found:
        u = u.rstrip(".,;|)")
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned

def safe_filename(url: str, index: int) -> str:
    """Generate a safe, unique filename for a downloaded URL.

    Extracts the domain (netloc) and last path segment from the URL, sanitizes
    them to filesystem-safe characters, truncates long segments, and appends a
    hash of the URL for uniqueness. Result format: "01_netloc_pathseg_hash.html".

    Args:
        url (str): The URL to generate a filename for.
        index (int): A 1-based ordinal index (used as the leading number).

    Returns:
        str: A filesystem-safe filename string (e.g., "01_www.reddit.com_comments_abc123.html").

    Example:
        >>> safe_filename("https://www.reddit.com/r/Penn/comments/xyz", 1)
        '01_www.reddit.com_comments_4a5f9e2b.html'
    """
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/")
    last = path.split("/")[-1] if path else "root"
    last = re.sub(r"[^A-Za-z0-9._-]", "_", last)
    if len(last) > 30:
        last = last[:30]
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    filename = f"{index:02d}_{netloc}_{last}_{h}.html"
    return filename

def fetch_with_requests(url: str, headers: dict, timeout: int = 15, tries: int = 3) -> tuple:
    """Fetch a URL using the requests library with retry logic.

    Attempts to fetch the URL up to `tries` times, with a 1-second delay between
    retries. Returns success status, response body, and HTTP status code.

    Args:
        url (str): The URL to fetch.
        headers (dict): HTTP headers dict (e.g., {"User-Agent": "..."}).
        timeout (int, optional): Request timeout in seconds. Defaults to 15.
        tries (int, optional): Maximum number of retry attempts. Defaults to 3.

    Returns:
        tuple: A 3-tuple (success, body, status_code):
            - success (bool): True if a 200 response was received.
            - body (str or None): Response body if successful, None otherwise.
            - status_code (int or None): HTTP status code, or None if unreachable.

    Example:
        >>> ok, body, code = fetch_with_requests("https://...", {"User-Agent": "..."})
        >>> if ok:
        ...     print(len(body), "bytes retrieved")
    """
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.text, resp.status_code
            else:
                print(f"Warning: {url} returned status {resp.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e} (attempt {attempt}/{tries})")
        time.sleep(1)
    return False, None, None

def fetch_with_playwright(url: str, browser_name: str = "chromium", timeout: int = 30,
                          interactive: bool = False, user_agent: str | None = None) -> tuple:
    """Fetch a URL using Playwright (for JS-heavy pages).

    Launches a headless (or headful if interactive=True) browser, navigates to the
    URL, optionally clicks consent buttons and scrolls (for SPA content rendering),
    and returns the page HTML. Handles navigation errors gracefully.

    Args:
        url (str): The URL to fetch.
        browser_name (str, optional): Browser type ("chromium", "firefox", "webkit").
                                      Defaults to "chromium".
        timeout (int, optional): Navigation timeout in seconds. Defaults to 30.
        interactive (bool, optional): Launch in headful mode and attempt cookie
                                      consent interactions. Defaults to False.
        user_agent (str, optional): Custom user agent string. Defaults to None.

    Returns:
        tuple: A 3-tuple (success, content, status_code):
            - success (bool): True if page content was retrieved.
            - content (str or None): The page HTML if successful, None otherwise.
            - status_code (int or None): 200 if successful, None otherwise.

    Raises:
        None: Returns (False, None, None) on any Playwright error.
    """
    if sync_playwright is None:
        print("Playwright not installed; cannot fetch with browser.")
        return False, None, None
    try:
        with sync_playwright() as p:
            browser = getattr(p, browser_name).launch(headless=not interactive)
            context_args = {}
            if user_agent:
                context_args["user_agent"] = user_agent
            context_args["viewport"] = {"width": 1280, "height": 800}
            context = browser.new_context(**context_args)
            page = context.new_page()
            page.set_default_navigation_timeout(timeout * 1000)
            try:
                page.goto(url, wait_until="networkidle")
            except Exception:
                try:
                    page.goto(url)
                except Exception as e:
                    print(f"Playwright navigation error for {url}: {e}")
            if interactive:
                try:
                    selectors = [
                        "button:has-text(\"Accept\")",
                        "button:has-text(\"I agree\")",
                        "button:has-text(\"I Agree\")",
                        "button:has-text(\"Agree\")",
                        "button:has-text(\"Accept Cookies\")",
                        "text=Accept Cookies",
                        "text=I agree",
                    ]
                    for sel in selectors:
                        try:
                            locator = page.locator(sel)
                            if locator.count() > 0:
                                locator.first.click(timeout=2000)
                                page.wait_for_timeout(500)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1500)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(500)
                except Exception:
                    pass
            content = page.content()
            page.close()
            context.close()
            browser.close()
            return True, content, 200
    except Exception as e:
        print(f"Playwright fetch error for {url}: {e}")
        return False, None, None

def main() -> None:
    """Parse arguments and download documents from URLs.

    Extracts URLs from a markdown file, fetches each URL (with optional Playwright
    fallback), saves HTML files with sequential numbering, and records metadata
    (success, status codes, file sizes) in a JSON file.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="planning.md", help="Markdown file with URLs (planning.md)")
    p.add_argument("--out", default="documents", help="Output directory for saved HTML files")
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--user-agent", default="Mozilla/5.0 (compatible; RAG-Document-Pipeline/1.0)")
    p.add_argument("--use-playwright", action="store_true", help="Fallback to Playwright for JS-heavy pages")
    p.add_argument("--playwright-interactive", action="store_true", help="Launch Playwright in interactive (headful) mode and attempt cookie clicks/scrolling")
    p.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = extract_urls_from_markdown(src)
    if not urls:
        print("No URLs found in", src)
        return

    headers = {"User-Agent": args.user_agent}
    results = []
    # Iterate through the extracted URLs, attempt to fetch each one using requests (with optional fallback to Playwright)
    # Save the content to HTML files, and record metadata about each fetch attempt.
    for i, url in enumerate(urls, start=1):
        print(f"Fetching ({i}/{len(urls)}): {url}")
        ok, body, status = fetch_with_requests(url, headers, timeout=args.timeout, tries=args.tries)
        if not ok and (args.use_playwright or args.playwright_interactive):
            print("Falling back to Playwright...")
            ok, body, status = fetch_with_playwright(url, browser_name=args.browser, timeout=args.timeout,
                                                     interactive=args.playwright_interactive, user_agent=args.user_agent)
        # Generate a safe filename for the URL, save the content if fetched successfully, and record the result in the metadata list.
        filename = safe_filename(url, i)
        out_path = out_dir / filename
        item = {"url": url, "file": str(out_path), "ok": ok, "status_code": status}
        if ok and body:
            out_path.write_text(body, encoding="utf-8")
            item["size"] = len(body)
            print(f"Saved -> {out_path}")
        else:
            print(f"Failed to fetch: {url} (status: {status})")
        results.append(item)
        time.sleep(args.sleep)

    # Save the metadata about each fetch attempt to a JSON file in the output directory, and print a summary of the download results.
    meta_path = out_dir / "download_metadata.json"
    meta_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"Downloaded {ok_count}/{len(results)} pages. Metadata: {meta_path}")

# Entry point for the script
if __name__ == "__main__":
    main()
