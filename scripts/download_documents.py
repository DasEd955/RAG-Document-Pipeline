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
    text = md_path.read_text(encoding="utf-8")
    # crude URL regex that skips trailing table delimiters
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
    for i, url in enumerate(urls, start=1):
        print(f"Fetching ({i}/{len(urls)}): {url}")
        ok, body, status = fetch_with_requests(url, headers, timeout=args.timeout, tries=args.tries)
        if not ok and (args.use_playwright or args.playwright_interactive):
            print("Falling back to Playwright...")
            ok, body, status = fetch_with_playwright(url, browser_name=args.browser, timeout=args.timeout,
                                                    interactive=args.playwright_interactive, user_agent=args.user_agent)

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

    meta_path = out_dir / "download_metadata.json"
    meta_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"Downloaded {ok_count}/{len(results)} pages. Metadata: {meta_path}")


if __name__ == "__main__":
    main()
