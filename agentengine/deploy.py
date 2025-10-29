from google.auth import default
import vertexai

from agent.agent import picker

_, PROJECT = default()
LOCATION = 'us-central1'
BUCKET = f'gs://{PROJECT}'
REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]",
    "cloudpickle",
    "pydantic"
]

vertexai.init(project = PROJECT, location = LOCATION, staging_bucket=BUCKET)

remote_agent = vertexai.agent_engines.create(
    agent_engine = picker,
    display_name = 'Local deployment',
    requirements=REQUIREMENTS,
)