import vertexai
from vertexai._genai.types import CreateAgentEngineConfig
import streamlit as st

ENGINE = "projects/471578722063/locations/us-central1/reasoningEngines/2561435482206502912"

client = vertexai.Client(
  project="btvaroska",
  location="us-central1"
)

def create_new_session(name: str):
    config = CreateAgentEngineConfig(display_name = name, wait_for_completion=True)
    client.agent_engines.sessions.create(
        name=ENGINE,
        user_id = 'web',
        config = config
    )

create_new_session('test')

for session in client.agent_engines.sessions.list(
    name=ENGINE,  # Required
#    config={"filter": "user_id=USER_ID"},
):
    print(session)


# # Set page configuration
# st.set_page_config(
#     page_title="Basic Streamlit App",
#     layout="wide"
# )

# # Sidebar with list of items
# st.sidebar.title("Sessions")
# items = [f"Item {i}" for i in range(1, 6)]

# for item in items:
#     st.sidebar.write(item)

# # Main content
# st.title("Basic Streamlit Application")
# st.write("Welcome to the basic Streamlit application!")
# st.write("Check the sidebar on the left to see the list of items.")
