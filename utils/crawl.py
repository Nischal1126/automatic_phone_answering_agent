import asyncio
import re
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

# Where PDFs (and markdown, if you want it) get saved
OUTPUT_DIR = Path("doai_output")
OUTPUT_DIR.mkdir(exist_ok=True)


filter_chain = FilterChain([
    DomainFilter(allowed_domains=["doai.ku.edu.np"]),
    URLPatternFilter(
        patterns=["*academic-activities*", "*gallery*"],
        reverse=True,  # reverse=True => BLOCK matches, allow everything else
    ),
])


def safe_filename(url: str) -> str:
    """Turn a URL into a filesystem-safe filename."""
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return name[:150] or "page"


async def main():
    browser_conf = BrowserConfig(
        headless=True,
        user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        ),
        browser_type="chromium",
        browser_mode="dedicated",
    )

    run_conf = CrawlerRunConfig(
        pdf=True,
        exclude_all_images=True,
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=True,
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3,
            filter_chain=filter_chain,
        ),
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun(
            url="https://doai.ku.edu.np",
            config=run_conf,
        )

        # results is a LIST when deep_crawl_strategy is set
        print(f"Crawled {len(results)} pages")

        for result in results:
            if not result.success:
                print(f"[FAILED] {result.url}: {result.error_message}")
                continue

            fname = safe_filename(result.url)

            # Save markdown per page
            if result.markdown:
                (OUTPUT_DIR / f"{fname}.md").write_text(
                    str(result.markdown), encoding="utf-8"
                )

            # Save PDF per page
            if result.pdf:
                (OUTPUT_DIR / f"{fname}.pdf").write_bytes(result.pdf)
                print(f"[OK] Saved PDF for {result.url}")
            else:
                print(f"[WARN] No PDF for {result.url}")


if __name__ == "__main__":
    asyncio.run(main())