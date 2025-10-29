import sys
import vertexai
from google.auth import default

from agent.agent import second_picker

# Engine name
if len(sys.argv) < 2:
    print('Using hardcoded engine id')
    engine_name = '2076568446543331328'
else:
    engine_name = sys.argv[1]

_, PROJECT = default()
LOCATION = 'us-central1'
USER_ID = 'tester'
BUCKET = f'gs://{PROJECT}'
REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]",
    "cloudpickle",
    "pydantic"
]

vertexai.init(
    project=PROJECT,
    location=LOCATION,
    staging_bucket=BUCKET
)

vertexai.agent_engines.update(
    resource_name = f"projects/{PROJECT}/locations/{LOCATION}/reasoningEngines/{engine_name}",
    agent_engine = second_picker
)