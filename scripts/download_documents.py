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

# extract_urls_from_markdown(): Function that takes a Path to a markdown file, reads its content, and uses a regex to extract and clean URLs
    # Cleaning involves stripping trailing punctuation and ensuring uniqueness. It returns a list of unique URLs found in the markdown text.
    # Returns a list of unique URLs found in the markdown text.
def extract_urls_from_markdown(md_path: Path) -> list:
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

# safe_filename(): Function that generates a safe filename for a given URL and index. 
    # It parses the URL, extracts the netloc and last path segment, sanitizes them, and combines them with a hash of the URL to create a unique filename.
def safe_filename(url: str, index: int) -> str:
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

# fetch_with_requests(): Function that attempts to fetch a URL using the requests library, with specified headers, timeout, and retry logic.
    # It returns a tuple indicating success, the response text (if successful), and the status code.
def fetch_with_requests(url: str, headers: dict, timeout: int = 15, tries: int = 3):
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

# fetch_with_playwright(): Function that attempts to fetch a URL using Playwright, with options for browser type, timeout, interactive mode, and user agent.
    # It handles navigation, optional interactions (like clicking cookie consent buttons), and returns a tuple indicating success, the page content (if successful), and a status code (200 if content is retrieved, None otherwise). 
    # It also includes error handling for Playwright operations.
def fetch_with_playwright(url: str, browser_name: str = "chromium", timeout: int = 30,
                          interactive: bool = False, user_agent: str | None = None):
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

# main(): Command-line interface function that parses arguments for input markdown file, output directory, timeout, retry attempts, user agent, and Playwright options.
    # It extracts URLs from the specified markdown file, attempts to fetch each URL using requests (with optional fallback to Playwright)
    # Saves the content to HTML files in the output directory, and records metadata about each fetch attempt in a JSON file. Finally, it prints a summary of the download results.
def main():
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
