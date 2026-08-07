import argparse
import asyncio
import codecs
import json
import re
from pathlib import Path

import fnmatch

import aiohttp
import pdfplumber
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, URLPatternFilter


def is_document_link(href: str, doc_domains: list[str], doc_url_patterns: list[str]) -> bool:
    """A link counts as a 'document to download', not a 'page to crawl', if
    it's hosted on a known file domain (e.g. cdn.ku.edu.np) OR matches a
    known file-serving URL pattern on the *same* domain (e.g. a site's own
    '/file-access/<id>' download route, which isn't a separate domain and
    so DomainFilter alone won't catch it)."""
    if any(d in href for d in doc_domains):
        return True
    return any(fnmatch.fnmatch(href, p) for p in doc_url_patterns)

# Known magic-byte signatures -> (extension, content type label)
MAGIC_SIGNATURES = [
    (b"%PDF-", ".pdf", "pdf"),
    (b"PK\x03\x04", ".docx", "docx/zip"),  # docx/pptx/xlsx are all zip-based
]


def guess_extension(content: bytes, content_type: str) -> str:
    for sig, ext, _label in MAGIC_SIGNATURES:
        if content.startswith(sig):
            return ext
    if "pdf" in content_type:
        return ".pdf"
    if "word" in content_type or "officedocument" in content_type:
        return ".docx"
    return ".bin"


def try_rot13_reveal(url_path_segment: str) -> str | None:
    """If a path segment ROT13-decodes into something that looks like a
    real filename (contains a known doc extension), return the decoded
    name. Otherwise return None."""
    decoded = codecs.encode(url_path_segment, "rot13")
    if re.search(r"\.(pdf|docx?|pptx?|xlsx?)$", decoded, re.IGNORECASE):
        return decoded
    return None


def safe_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:max_len]


# ---------------------------------------------------------------------------
# Step 1: crawl pages and collect candidate document links
# ---------------------------------------------------------------------------
async def discover_document_links(
    start_url: str,
    site_domain: str,
    doc_domains: list[str],
    doc_url_patterns: list[str],
    max_depth: int = 3,
):
    filters = [DomainFilter(allowed_domains=[site_domain])]
    if doc_url_patterns:
        # Block the browser from ever trying to *navigate* to same-domain
        # file-serving routes (e.g. "*/file-access/*"). Without this, Chromium
        # tries to page.goto() a direct-download URL, the navigation aborts
        # with "Download is starting", and depending on how the download
        # event is handled that can stall or kill the whole crawl. Excluding
        # it here only stops it from being *visited* as a page — the link is
        # still recorded in result.links on the page that references it, so
        # our own downloader below still finds and fetches it separately.
        filters.append(URLPatternFilter(patterns=doc_url_patterns, reverse=True))

    filter_chain = FilterChain(filters)

    browser_conf = BrowserConfig(headless=True, browser_type="chromium")
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=False,  # IMPORTANT: keep external links so
        # doc-hosting CDN links (a different domain) still show up.
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=max_depth, filter_chain=filter_chain),
    )

    doc_links = {}  # href -> {text, found_on}
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun(url=start_url, config=run_conf)
        for result in results:
            if not result.success:
                continue
            all_links = result.links.get("internal", []) + result.links.get("external", [])
            for link in all_links:
                href = link.get("href", "")
                if is_document_link(href, doc_domains, doc_url_patterns):
                    doc_links.setdefault(href, {"text": link.get("text", ""), "found_on": result.url})

    return doc_links


# ---------------------------------------------------------------------------
# Step 2: download + verify + extract
# ---------------------------------------------------------------------------
async def download_and_extract(session: aiohttp.ClientSession, url: str, meta: dict, output_dir: Path):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            content = await resp.read()
            content_type = resp.headers.get("Content-Type", "").lower()
    except Exception as e:
        print(f"[FAIL] {url} -> {e}")
        return None

    ext = guess_extension(content, content_type)
    if ext == ".bin":
        print(f"[SKIP] {url} - not a recognized document type ({content_type})")
        return None

    # Try to recover the real filename from ROT13-obfuscated path segments
    revealed_name = None
    for segment in url.rstrip("/").split("/"):
        revealed = try_rot13_reveal(segment)
        if revealed:
            revealed_name = revealed
            break

    if revealed_name:
        base_name = safe_filename(Path(revealed_name).stem)
    else:
        base_name = safe_filename(Path(url.split("?")[0]).stem) or "document"

    pdf_path = output_dir / f"{base_name}{ext}"
    pdf_path.write_bytes(content)

    md_text = None
    if ext == ".pdf":
        try:
            with pdfplumber.open(pdf_path) as pdf:
                md_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip()
        except Exception as e:
            print(f"[WARN] Could not extract text from {pdf_path.name}: {e}")

    if md_text:
        (output_dir / f"{base_name}.md").write_text(md_text, encoding="utf-8")

    print(f"[OK] {url}\n     -> {pdf_path.name}"
          + (f" (+ .md, {len(md_text)} chars)" if md_text else ""))

    return {
        "source_url": url,
        "local_file": pdf_path.name,
        "found_on_page": meta["found_on"],
        "link_text": meta["text"],
        "revealed_filename": revealed_name,
    }


async def run(
    start_url: str,
    site_domain: str,
    doc_domains: list[str],
    doc_url_patterns: list[str],
    output: str,
    max_depth: int,
):
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Crawling {start_url} (domain={site_domain}, depth={max_depth}) for document links "
          f"on domains={doc_domains} patterns={doc_url_patterns} ...")
    doc_links = await discover_document_links(start_url, site_domain, doc_domains, doc_url_patterns, max_depth)
    print(f"Found {len(doc_links)} candidate document links.")

    manifest = []
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        for url, meta in doc_links.items():
            record = await download_and_extract(session, url, meta, output_dir)
            if record:
                manifest.append(record)
            await asyncio.sleep(1)  # be polite / rate-limit

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} documents saved to {output_dir}/  (manifest: {manifest_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_url", required=True, help="Page to crawl for document links, e.g. course listing page")
    parser.add_argument("--site_domain", default=None, help="Domain to restrict page-crawling to (defaults to start_url's domain)")
    parser.add_argument("--doc_domains", nargs="+", default=[], help="Domain(s) that host the actual documents, e.g. cdn.ku.edu.np")
    parser.add_argument("--doc_url_patterns", nargs="+", default=[],
                         help="Glob pattern(s) for same-domain file-serving routes, e.g. '*/file-access/*'")
    parser.add_argument("--output", default="course_docs")
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    if not args.doc_domains and not args.doc_url_patterns:
        parser.error("Provide at least one of --doc_domains or --doc_url_patterns")

    from urllib.parse import urlparse
    site_domain = args.site_domain or urlparse(args.start_url).netloc

    asyncio.run(run(args.start_url, site_domain, args.doc_domains, args.doc_url_patterns, args.output, args.max_depth))