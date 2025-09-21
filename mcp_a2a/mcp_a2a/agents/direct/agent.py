from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

root_agent = LlmAgent(
    name="currency_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to manage currency conversions"
    ),
    instruction=(
        "You are a helpful agent who can calculate exchange rates between currencies"
    ),
    tools=[MCPToolset(
        connection_params=SseServerParams(
            url="http://localhost:8080/sse",
        )
    )],
)
