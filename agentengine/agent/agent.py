from google.adk.agents import LlmAgent

MODEL = 'gemini-2.5-flash'
PROMPT = """
Create hypothetical recoomendation for stock selection. Use atock sticker which does not exists.
"""
picker = LlmAgent(
    name = 'picker',
    model = MODEL,
    instruction=PROMPT,
    description='Stock picker agent'
)

second_picker = LlmAgent(
    name = 'picker',
    model = MODEL,
    instruction="Refuse answer any question. Refer user to the helpdesk",
    description='Stock picker agent'
)

root_agent = picker