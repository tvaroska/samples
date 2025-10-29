import subprocess
import cloudpickle
import vertexai

from agent.agent import picker

ROOT_DIR = 'terraform'
AGENT_FILE = 'agent.pkl'
REQUIREMENTS_FILE = 'requirements.txt'
REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]",
    "cloudpickle",
    "pydantic"
]

local_agent = vertexai.agent_engines.AdkApp(agent=picker)
with open(f'{ROOT_DIR}/{AGENT_FILE}', "wb") as f:
  cloudpickle.dump(local_agent, f)

with open(f'{ROOT_DIR}/{REQUIREMENTS_FILE}', "w") as f:
  f.writelines(REQUIREMENTS)
#  f.write("\n".join(REQUIREMENTS))

p = subprocess.Popen(
  ["tar", "-czf", "empty.tar.gz", "--files-from", "/dev/null"],
  cwd=ROOT_DIR
)
p.wait()
