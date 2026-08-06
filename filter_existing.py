"""Check which articles from a title list already exist on the target wiki
(sq.wikipedia) via Wikidata sitelinks, and write only the missing ones."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 50


def normalize_title_case(title: str) -> str:
    if not title:
        return title
    return title[0].upper() + title[1:]


async def check_titles_exist(titles: list[str], source_lang: str, target_lang: str) -> dict[str, str | None]:
    """Returns {title: target_wiki_title_or_None}. Uses Wikidata sitelinks."""
    source_site = f"{source_lang}wiki"
    target_site = f"{target_lang}wiki"

    unique = list(dict.fromkeys(titles))
    norm_to_original: dict[str, str] = {}
    for original in unique:
        norm_to_original.setdefault(normalize_title_case(original), original)
    normalized = list(norm_to_original.keys())

    result: dict[str, str | None] = {}
    for i in range(0, len(normalized), BATCH_SIZE):
        batch = normalized[i : i + BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "sites": source_site,
            "titles": "|".join(batch),
            "props": "sitelinks",
            "sitefilter": f"{source_site}|{target_site}",
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(
            headers={"User-Agent": "wiki-translate-harness/0.1.0 (adobroshi@limakkosovo.aero)"},
            timeout=30,
        ) as client:
            resp = await client.get(WIKIDATA_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        for entity in data.get("entities", {}).values():
            if "missing" in entity:
                continue
            sitelinks = entity.get("sitelinks", {})
            source_link = sitelinks.get(source_site)
            if not source_link:
                continue
            target_link = sitelinks.get(target_site)
            target_title = target_link["title"] if target_link else None
            # Map back to original casing
            norm = normalize_title_case(source_link["title"])
            if norm in norm_to_original:
                result[norm_to_original[norm]] = target_title

    # Titles not found in Wikidata at all are treated as missing
    for title in unique:
        if title not in result:
            result[title] = None

    return result


async def main() -> None:
    titles_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("albanian_mythology")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else titles_file.with_suffix(".missing")
    source_lang = "en"
    target_lang = "sq"

    titles = [
        line.strip()
        for line in titles_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    print(f"Checking {len(titles)} titles against {target_lang}.wikipedia.org...")
    checks = await check_titles_exist(titles, source_lang, target_lang)

    existing: list[tuple[str, str]] = []
    missing: list[str] = []
    for title in titles:
        target = checks.get(title)
        if target is not None:
            existing.append((title, target))
        else:
            missing.append(title)

    print(f"\n  Already exist on sq.wikipedia: {len(existing)}")
    for title, target in existing:
        print(f"    {title} -> {target}")

    print(f"\n  Missing (will translate): {len(missing)}")
    for title in missing:
        print(f"    {title}")

    output_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"\nWritten to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
