from dotenv import load_dotenv
from tavily import TavilyClient
import os

load_dotenv()

client = TavilyClient(
    api_key = os.getenv("TAVILY_API_KEY")
)

def tavily_search(query: str) -> str:
    """Performs a web search using the Tavily API and formats the results.

    Args:
        query (str): The search term or question to search the web for.

    Returns:
        str: A formatted string containing the top 3 search results, 
             including the title, URL, and a snippet of the content for each.
    """

    respose = client.search(
        query=query,
        max_results=3
    )

    results = []

    for i, r in enumerate(respose["results"], 1):
        title = r.get('title', "Unknown")
        url = r.get('url', '')
        snippet = r.get('content', '').strip()

        # Keep only the first 300 characters to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)


