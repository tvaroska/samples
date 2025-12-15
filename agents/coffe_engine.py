"""
Create agent engine for Coffe Agent
"""
import uuid
import datetime

import vertexai

from google.auth import default as get_project

from coffee.agent import root_agent

MemoryBankConfig = vertexai.types.ReasoningEngineContextSpecMemoryBankConfig
SimilaritySearchConfig = vertexai.types.ReasoningEngineContextSpecMemoryBankConfigSimilaritySearchConfig
GenerationConfig = vertexai.types.ReasoningEngineContextSpecMemoryBankConfigGenerationConfig

MemoryTopic = vertexai.types.MemoryBankCustomizationConfigMemoryTopic
CustomMemoryTopic = vertexai.types.MemoryBankCustomizationConfigMemoryTopicCustomMemoryTopic
CustomizationConfig = vertexai.types.MemoryBankCustomizationConfig


_, project_id = get_project()
LOCATION = 'us-central1'
REQUIREMENTS = [
    "google-adk<1.18.0",
]


vertexai.init(project=project_id, location=LOCATION, staging_bucket=f'gs://{project_id}')
client = vertexai.Client(
    project = project_id,
    location = LOCATION    
)

custom_coffe_topics = [
    MemoryTopic(
        custom_memory_topic=CustomMemoryTopic(
            label="Type of coffee",
            description="""Extract the client's typical coffe order. Example: Espresso, single shot""",
        )
    ),
    MemoryTopic(
        custom_memory_topic=CustomMemoryTopic(
            label="Payment type",
            description="""Extract the client's usual payment type. Examples: cash, credit card""",
        )
    )
]

memory_config = MemoryBankConfig(
    # Embedding model for similarity search
    similarity_search_config=SimilaritySearchConfig(
        embedding_model=f"projects/{project_id}/locations/{LOCATION}/publishers/google/models/text-embedding-005"
    ),
    # LLM for extracting memories from conversations
    generation_config=GenerationConfig(
        model=f"projects/{project_id}/locations/{LOCATION}/publishers/google/models/gemini-2.5-flash"
    ),
    customization_configs = [
        CustomizationConfig(memory_topics=custom_coffe_topics),
    ]
)

default_agent_engine = client.agent_engines.create(
#    agent = root_agent,
    staging_bucket=f'gs://{project_id}',
    config={
        "display_name": "Barista",
        "context_spec": {"memory_bank_config": memory_config}
    }
)

default_engine_name = default_agent_engine.api_resource.name

print(f"Agent Engine Name: {default_engine_name}\n")

client_id = "client_" + str(uuid.uuid4())[:4]
session = client.agent_engines.sessions.create(
    name=default_engine_name,
    user_id=client_id,
    config={"display_name": f"Order for {client_id}"},
)
session_name = session.response.name

first_order = [
    {
        "role": "user",
        "message": "Hi, single espreso pls.",
    },
    {
        "role": "model",
        "message": "Thank you, that will be 1 dollar. Anything else?",
    },
    {
        "role": "user",
        "message": "That is all.",
    },
    {
        "role": "model",
        "message": "How do you want to pay?",
    },
    {
        "role": "user",
        "message": "Cash",
    },
    {
        "role": "model",
        "message": "Your order number is 42. Thank you for your business",
    }
]

invocation_id = 0

for turn in first_order:
    client.agent_engines.sessions.events.append(
        name=session_name,
        author=client_id,  # Required: who is speaking
        invocation_id=str(invocation_id),  # Required: conversation sequence
        timestamp=datetime.datetime.now(tz=datetime.timezone.utc),  # Required: when
        config={
            "content": {"role": turn["role"], "parts": [{"text": turn["message"]}]}
        },
    )

    invocation_id += 1
    icon = "👤" if turn["role"] == "user" else "🤖"
    print(f"{icon} {turn['message']}")

print("\n✅ Conversation added to session successfully!")
print("💡 Now let's see what memories are extracted with default topics...")

default_operation = client.agent_engines.memories.generate(
    name=default_engine_name,
    vertex_session_source={"session": session_name},
    config={"wait_for_completion": True},  # Wait for completion (blocking call)
)
