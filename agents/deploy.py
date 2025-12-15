from google.auth import default
import vertexai

#from picker.agent import root_agent
from coffee.agent import root_agent

_, PROJECT = default()
LOCATION = 'us-central1'
BUCKET = f'gs://{PROJECT}'
REQUIREMENTS = [
    "google-adk<1.18.0",
]

vertexai.init(project = PROJECT, location = LOCATION, staging_bucket=BUCKET)

remote_agent = vertexai.agent_engines.create(
    agent_engine = root_agent,
    display_name = 'Barista',
    requirements=REQUIREMENTS,
)
