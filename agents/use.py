import sys
import asyncio
import vertexai
from google.auth import default


# Engine name
if len(sys.argv) < 2:
    print('Using hardcoded engine id')
    engine_name = '2561435482206502912'
else:
    engine_name = sys.argv[1]

_, PROJECT = default()
LOCATION = 'us-central1'
USER_ID = 'tester'

async def main():
    client = vertexai.Client(
        project=PROJECT,
        location=LOCATION,
    )

    engine = client.agent_engines.get(name=f"projects/{PROJECT}/locations/{LOCATION}/reasoningEngines/{engine_name}")
    async for event in engine.async_stream_query(
        user_id="USER_ID",
        # session_id="SESSION_ID",  # Optional
        message="What is the exchange rate from US dollars to SEK today?",
    ):
        print(event)

asyncio.run(main())