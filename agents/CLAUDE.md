# Claude Code Guide for Google ADK Projects

## Quick Reference for ADK Development

This guide provides essential information for working with Google's Agent Development Kit (ADK) using Claude Code.

---

## Project Setup

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install ADK
pip install google-adk
```

### Environment Configuration
Create `.env` file:
```bash
# For Google AI Studio
GOOGLE_API_KEY=your_api_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# For Vertex AI
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

### Local Development Commands
```bash
adk web          # Browser-based Dev UI
adk run          # Terminal interaction
adk api_server   # Local FastAPI server
adk eval         # Run evaluations
```

---

## Core Agent Types

### 1. LLM Agents (`LlmAgent`)
Use LLMs for reasoning, planning, and dynamic decision-making.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name or zip code

    Returns:
        Dictionary with temperature and conditions
    """
    return {"temperature": "72°F", "conditions": "sunny"}

agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash",
    description="Provides weather information",
    instruction="You are a helpful weather assistant.",
    tools=[FunctionTool(get_weather)]
)
```

### 2. Workflow Agents (Deterministic Control)

**SequentialAgent** - Execute agents in order:
```python
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="data_pipeline",
    sub_agents=[data_collector, data_processor, data_analyzer]
)
```

**ParallelAgent** - Execute agents concurrently:
```python
from google.adk.agents import ParallelAgent

parallel = ParallelAgent(
    name="multi_source",
    sub_agents=[news_agent, social_agent, blog_agent]
)
```

**LoopAgent** - Iterative execution:
```python
from google.adk.agents import LoopAgent

loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[generator, reviewer],
    max_iterations=3
)
```

---

## Tools Development

### Function Tool Best Practices

```python
from google.adk.tools import FunctionTool, ToolContext

def search_database(query: str, limit: int, tool_context: ToolContext) -> dict:
    """Search the product database.

    Args:
        query: Search keywords
        limit: Maximum number of results (1-100)

    Returns:
        Dictionary with 'status', 'results', and 'count' keys
    """
    # Access session state
    user_id = tool_context.state.get('user_id')

    # Perform search
    results = perform_search(query, limit, user_id)

    # Update state
    tool_context.state['last_search'] = query

    return {
        "status": "success",
        "results": results,
        "count": len(results)
    }
```

**Key Guidelines:**
- Use descriptive verb-noun names (e.g., `get_weather`, `search_documents`)
- Return `dict` (Python) or `Map` (Java)
- Include type hints for all parameters
- No default parameter values
- Write comprehensive docstrings
- Keep tools focused on single tasks
- Use JSON-serializable types only

### Tool Types

**Built-in Tools:**
- `google_search` - Web search (Gemini 2 models)
- `built_in_code_execution` - Execute code safely
- `vertex_ai_search` - Search private data stores
- `GkeCodeExecutor` - Sandboxed code execution in GKE

**Third-Party Tools:**
```python
from google.adk.tools import LangchainTool, CrewaiTool
from langchain.tools.tavily_search import TavilySearchResults

search = LangchainTool(TavilySearchResults())
```

**Agent as Tool:**
```python
from google.adk.tools import AgentTool

specialist = LlmAgent(name="specialist", ...)
coordinator = LlmAgent(
    name="coordinator",
    tools=[AgentTool(specialist)]
)
```

**OpenAPI Tools:**
```python
from google.adk.toolsets import OpenAPIToolset

toolset = OpenAPIToolset(
    openapi_spec_path="path/to/spec.yaml",
    auth_config={"type": "api_key", "api_key": "KEY"}
)
```

**MCP Tools:**
```python
from google.adk.toolsets import MCPToolset

mcp = MCPToolset(
    server_command=["uvx", "mcp-server-sqlite"],
    server_params={"db_path": "data.db"}
)
```

---

## Multi-Agent Patterns

### 1. Coordinator/Dispatcher
```python
coordinator = LlmAgent(
    name="coordinator",
    description="Routes requests to specialists",
    sub_agents=[sales_agent, support_agent, billing_agent]
)
# LLM uses transfer_to_agent() for routing
```

### 2. Sequential Pipeline
```python
pipeline = SequentialAgent(
    name="content_pipeline",
    sub_agents=[researcher, writer, editor]
)
# Uses session.state to pass data between stages
```

### 3. Generator-Critic Pattern
```python
review_loop = LoopAgent(
    name="review_loop",
    sub_agents=[
        LlmAgent(name="generator", output_key="draft"),
        LlmAgent(name="critic", output_key="feedback")
    ],
    max_iterations=3
)
```

### 4. Parallel Fan-Out/Gather
```python
parallel = ParallelAgent(
    name="multi_source",
    sub_agents=[source1, source2, source3]
)
aggregator = LlmAgent(
    name="aggregator",
    instruction="Synthesize results from {combined_data}"
)
system = SequentialAgent(sub_agents=[parallel, aggregator])
```

---

## State Management

### Session State (Temporary)
```python
# In tool function
def my_tool(tool_context: ToolContext) -> dict:
    # Read state
    user_pref = tool_context.state.get('preference')

    # Write state (auto-tracked)
    tool_context.state['last_action'] = 'search'

    return {"status": "ok"}

# In agent instruction (templating)
agent = LlmAgent(
    instruction="User preference is {preference}. Last action: {last_action?}"
)
```

**State Scopes:**
- `session_key` - Current session only
- `user:key` - Across user's sessions
- `app:key` - Application-wide
- `temp:key` - Not persisted

### Using output_key
```python
agent = LlmAgent(
    name="analyzer",
    output_key="analysis_result"  # Auto-saves response to state
)
```

### Memory (Long-term)
```python
from google.adk.services import VertexAiRagMemoryService

memory_service = VertexAiRagMemoryService(corpus_name="user_history")

# In tool
def recall_info(query: str, tool_context: ToolContext) -> dict:
    results = tool_context.search_memory(query)
    return {"memories": results}
```

---

## Callbacks for Control

### Guardrails and Validation
```python
def before_tool_callback(context: CallbackContext, tool_name: str, args: dict):
    """Validate tool calls before execution."""

    # Block sensitive operations
    if tool_name == "delete_data" and not context.state.get('admin'):
        return {"error": "Unauthorized", "blocked": True}

    # Validate arguments
    if tool_name == "transfer_funds":
        if args.get('amount', 0) > 10000:
            return {"error": "Amount exceeds limit", "blocked": True}

    # Allow execution
    return None

agent = LlmAgent(
    name="banking_agent",
    tools=[...],
    before_tool_callback=before_tool_callback
)
```

### Logging and Monitoring
```python
def after_model_callback(context: CallbackContext, response):
    """Log LLM interactions."""
    logger.info(f"Agent: {context.agent_name}, Tokens: {response.usage}")
    return None  # Don't modify response
```

### Caching
```python
def before_model_callback(context: CallbackContext, messages):
    """Check cache before LLM call."""
    cache_key = hash(str(messages))
    cached = cache.get(cache_key)
    if cached:
        return cached  # Skip LLM call
    return None  # Proceed with LLM call
```

**Callback Types:**
- `before_agent` / `after_agent` - Agent execution
- `before_model` / `after_model` - LLM calls
- `before_tool` / `after_tool` - Tool execution

**Return Behavior:**
- `return None` - Continue normal flow
- `return <object>` - Override default behavior (skip execution)

---

## Deployment

### Vertex AI Agent Engine
```python
from vertexai.preview import reasoning_engines

app = reasoning_engines.AdkApp(agent=my_agent)

remote_app = reasoning_engines.ReasoningEngine.create(
    app,
    requirements=["google-adk"],
    display_name="my-agent"
)
```

### Cloud Run
```bash
adk deploy cloud_run \
    --project=PROJECT_ID \
    --region=us-central1 \
    --agent-path=path/to/agent.py
```

### GKE
```bash
# Build container
docker build -t gcr.io/PROJECT_ID/agent:latest .

# Deploy with kubectl
kubectl apply -f deployment.yaml
```

---

## Evaluation

### Test Files (.test.json)
```json
{
  "user_content": "What's the weather in NYC?",
  "expected_tool_use": [
    {"name": "get_weather", "args": {"location": "NYC"}}
  ],
  "expected_response": "sunny and 72 degrees"
}
```

### Evalset Files (Multi-turn)
```json
{
  "evals": [
    {
      "turns": [
        {
          "user_query": "Book a flight to Paris",
          "expected_tool_use": [{"name": "search_flights"}],
          "expected_response": "I found 3 flights"
        },
        {
          "user_query": "Select the cheapest",
          "expected_tool_use": [{"name": "book_flight"}],
          "reference_response": "Booked flight AF123"
        }
      ]
    }
  ]
}
```

### Run Evaluations
```bash
# Web UI
adk web  # Navigate to Evaluations tab

# CLI
adk eval --evalset=path/to/evalset.json

# pytest
pytest agent_test.py
```

**Metrics:**
- `tool_trajectory_avg_score` - Tool usage correctness (default: 1.0)
- `response_match_score` - Response quality via ROUGE (default: 0.8)

---

## Security Best Practices

### 1. Authentication
```python
# User Auth (OAuth)
from google.adk.auth import AuthScheme, AuthCredential

auth_config = AuthScheme(
    type="oauth2",
    credential=AuthCredential(
        client_id="CLIENT_ID",
        client_secret="SECRET"
    )
)
```

### 2. Input Validation
```python
def before_tool_callback(context, tool_name, args):
    # Sanitize inputs
    if 'sql_query' in args:
        if any(word in args['sql_query'].lower() for word in ['drop', 'delete']):
            return {"error": "Unsafe SQL"}
    return None
```

### 3. Sandboxed Code Execution
```python
from google.adk.tools import GkeCodeExecutor

executor = GkeCodeExecutor(
    cluster_name="my-cluster",
    namespace="code-execution"
)
```

### 4. Content Safety
```python
def before_model_callback(context, messages):
    """Check for harmful content."""
    user_input = messages[-1].content
    if contains_harmful_content(user_input):
        return create_blocked_response()
    return None
```

---

## Common Patterns

### Streaming Support
```python
agent = LlmAgent(
    model="gemini-2.0-flash-exp",  # Streaming-compatible
    tools=[...],
)

# Run with streaming
for event in agent.run_stream(user_input):
    if event.partial:
        print(event.content, end='', flush=True)
```

### Artifact Management
```python
def generate_report(tool_context: ToolContext) -> dict:
    # Create artifact
    report_content = generate_pdf_report()

    # Save artifact
    artifact_id = tool_context.save_artifact(
        content=report_content,
        filename="report.pdf",
        content_type="application/pdf"
    )

    return {"artifact_id": artifact_id}

def view_report(artifact_id: str, tool_context: ToolContext) -> dict:
    # Load artifact
    content = tool_context.load_artifact(artifact_id)
    return {"content": content}
```

### Dynamic Instructions
```python
def get_instruction(context: ReadonlyContext) -> str:
    """Generate instruction based on state."""
    role = context.state.get('user_role', 'guest')
    if role == 'admin':
        return "You have full access. Help with admin tasks."
    return "You have limited access. Help with basic queries."

agent = LlmAgent(
    instruction=get_instruction  # Function, not string
)
```

---

## Troubleshooting

### Common Issues

**Built-in Tools Limitation:**
- Only ONE built-in tool per root agent
- Cannot mix built-in tools with other tool types
- Built-in tools not supported in sub-agents

**State Not Persisting:**
- Ensure you're using `CallbackContext.state` or `ToolContext.state`
- Verify SessionService is configured correctly
- Check that state changes are in yielded Events

**Agent Not Transferring:**
- Ensure sub-agents have distinct descriptions
- Verify LLM generates `transfer_to_agent()` call
- Check agent hierarchy (parent-child relationships)

**Tool Not Being Called:**
- Improve docstring clarity and specificity
- Simplify parameter names and types
- Reduce number of available tools
- Make tool name more descriptive

---

## File Structure Example

```
my_agent_project/
├── .env                    # Environment configuration
├── __init__.py
├── agent.py               # Main agent definition
├── tools/
│   ├── __init__.py
│   ├── search.py
│   └── database.py
├── tests/
│   ├── agent.test.json
│   └── test_agent.py
├── evalsets/
│   └── evaluation.json
└── requirements.txt
```

---

## Key Takeaways

1. **Agent Types**: Use `LlmAgent` for reasoning, workflow agents for deterministic control
2. **Tools**: Return dicts, write clear docstrings, keep focused
3. **State**: Use `tool_context.state` for modifications, `output_key` for auto-save
4. **Multi-Agent**: Leverage patterns (coordinator, pipeline, parallel, loop)
5. **Callbacks**: Implement guardrails via `before_*` callbacks with return overrides
6. **Evaluation**: Test both trajectory and final response quality
7. **Security**: Validate inputs, use auth, sandbox code execution

---

## Additional Resources

- **Documentation**: https://cloud.google.com/vertex-ai/docs/adk
- **Python GitHub**: https://github.com/google/adk-python
- **Java GitHub**: https://github.com/google/adk-java
- **Community**: GitHub Discussions for Python/Java repos
