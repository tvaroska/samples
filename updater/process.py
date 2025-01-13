import asyncio
import json
import uuid
from datetime import datetime, timedelta
from time import mktime
from typing import List, Optional, Union

import feedparser
import httpx
from google import genai
from google.genai.errors import ClientError
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm.asyncio import tqdm

# Common globals
http_client = httpx.AsyncClient()


class Article(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    url: str
    title: str
    short_summary: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[Union[str, bytes]] = None
    mime_type: str


class Summary(BaseModel):
    short: str = Field(description="Tweet like summary of the article")
    long: str = Field(
        description="Summary of the article in form of independent text. Length of the text should be 250 words. Audience of this new version will consume it on the mobile phone during their commute. Respond in Markdown, each point as header3 and short support text for point."
    )


class ResourceExaused(ClientError):
    pass


semaphore = asyncio.Semaphore(10)
gemini = genai.Client(vertexai=True, location="us-central1")


async def get_article(entry, starting_point) -> Article:
    # Get published date from the entry
    if hasattr(entry, "published_parsed"):
        pub_date = datetime.fromtimestamp(mktime(entry.published_parsed))
    elif hasattr(entry, "updated_parsed"):
        pub_date = datetime.fromtimestamp(mktime(entry.updated_parsed))
    else:
        return None

    # Check if article is newer than starting_point
    if pub_date > starting_point:
        if hasattr(entry, "link"):
            # Determine media type based on feed URL and entry content
            # Check for youtube content
            media_type = None
            if "youtube.com" in entry.link:
                media_type = "video/vnd.youtube.yt"
            else:
                head = (await http_client.head(entry.link)).headers
                if "Content-Type" in head:
                    media_type = head["Content-Type"].split(";")[0]
        else:
            # No link - no article
            return None

        # Extract description as summary, handling potential missing field
        summary = None
        if hasattr(entry, "description"):
            summary = entry.description

        return Article(
            title=entry.title, url=entry.link, mime_type=media_type, summary=summary
        )
    else:
        return None


async def new_articles(feeds: List[str], starting_point: datetime) -> List[Article]:
    """
    Function will check RSS feeds in feeds and return all entries which are new from starting_point

    Args:
        feeds: List of RSS feed URLs to check
        starting_point: datetime object representing the cutoff point for new articles

    Returns:
        List of Article objects that were published after the starting_point
    """
    tasks = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)

        # Check if feed parsing was successful
        if "bozo_exception" in feed:
            print(f"Warning: Error parsing feed {feed_url}: {feed.bozo_exception}")
            continue

        tasks += [get_article(entry, starting_point) for entry in feed.entries]

    response = await asyncio.gather(*tasks)
    return [article for article in response if article]


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=10, jitter=5),
    retry=retry_if_exception_type(ResourceExaused),
)
async def get_summary(article: Article) -> Article:
    async with semaphore:
        try:
            response = await gemini.aio.models.generate_content(
                model="gemini-1.5-flash-002",
                contents=[
                    Part.from_text("Analyze the article"),
                    Part.from_uri(article.url, mime_type=article.mime_type),
                ],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Summary,
                ),
            )
        except ClientError as e:
            if e.code == 429:
                raise ResourceExaused(code=e.code, response=e.response) from e
            elif e.code == 400:
                # cannot load, page blocked for robots
                return None
            else:
                raise ClientError(code=e.code, response=e.response) from e
    summary = Summary.model_validate_json(response.text)

    article.short_summary = summary.short
    article.summary = summary.long

    return article


async def main():

    with open("settings.json") as f:
        data = json.load(f)

    starting_point = datetime.now() - timedelta(weeks=52)
    articles = await new_articles(data["feeds"], starting_point=starting_point)
    tasks = [get_summary(article) for article in articles]

    updated_articles = await tqdm.gather(*tasks)

    with open("articles.json", "w+") as f:
        json.dump(
            {
                "date": datetime.now().strftime("%b %d %Y"),
                "updates": [
                    {
                        "title": item.title,
                        "short": item.short_summary,
                        "long": item.summary,
                        "url": item.url,
                    }
                    for item in updated_articles
                    if item
                ],
            },
            f,
        )


if __name__ == "__main__":
    asyncio.run(main())
