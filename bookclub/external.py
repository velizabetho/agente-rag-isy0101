"""External sources: Open Library (book metadata) and Spanish Wikipedia (author context)."""

import requests

USER_AGENT = "bookclub-student-project/0.1 (educational use)"
TIMEOUT_S = 15
MAX_DESCRIPTION_CHARS = 800

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def _get_json(url: str, params: dict | None = None) -> dict:
    response = _session.get(url, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def search_open_library(title: str, author: str | None = None, limit: int = 3) -> list[dict]:
    """Search books by title/author; the best match includes the work description when available."""
    params = {
        "title": title,
        "limit": limit,
        "fields": "key,title,author_name,first_publish_year,subject,number_of_pages_median,language",
    }
    if author:
        params["author"] = author
    docs = _get_json("https://openlibrary.org/search.json", params).get("docs", [])

    results = []
    for i, doc in enumerate(docs):
        item = {
            "title": doc.get("title"),
            "authors": doc.get("author_name", [])[:3],
            "first_publish_year": doc.get("first_publish_year"),
            "median_pages": doc.get("number_of_pages_median"),
            "subjects": doc.get("subject", [])[:8],
            "url": f"https://openlibrary.org{doc['key']}",
        }
        if i == 0:
            work = _get_json(f"https://openlibrary.org{doc['key']}.json")
            description = work.get("description")
            if isinstance(description, dict):
                description = description.get("value")
            item["description"] = (description or "")[:MAX_DESCRIPTION_CHARS] or None
        results.append(item)
    return results


def wikipedia_summary(query: str) -> dict | None:
    """Best Spanish Wikipedia match for the query: title, summary and URL."""
    search = _get_json(
        "https://es.wikipedia.org/w/api.php",
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
    )
    hits = search.get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"].replace(" ", "_")
    summary = _get_json(f"https://es.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}")
    return {
        "title": summary.get("title"),
        "summary": summary.get("extract"),
        "url": summary.get("content_urls", {}).get("desktop", {}).get("page"),
    }
