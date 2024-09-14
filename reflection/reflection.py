"""

    Reflection subgraph

"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.base import RunnableSequence
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Generate the best essay possible for the user`s request."
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " If the user provides critique, respond with a new essay improved on feedback.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
writer_model = ChatVertexAI(model="gemini-1.5-flash-001")
default_writer = writer_prompt | writer_model

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reviewer_model = ChatVertexAI(model="gemini-1.5-flash-001")
default_reviewer = reviewer_prompt | reviewer_model

check_essay_prompt = PromptTemplate.from_template(
    "Check if the text is potenital talk about the paper.<TEXT>{essay}</TEXT>. Respond with Yes/No"
)
check_essay_model = VertexAI(model_name="gemini-1.5-flash-001")
default_check_essay = check_essay_prompt | check_essay_model


class ReflectionState(TypedDict):
    final: str
    previous: str
    counter: int
    messages: Annotated[AnyMessage, add_messages]


def reflection(
    writer: RunnableSequence = default_writer,
    reviewer: RunnableSequence = default_reviewer,
    check_essay: RunnableSequence = default_check_essay,
    max_count: int = 25,
) -> StateGraph:

    async def generation_node(state: ReflectionState) -> ReflectionState:
        if "counter" in state and state["counter"]:
            idx = state["counter"] + 1
        else:
            idx = 1

        response = {}
        response["counter"] = idx
        response["messages"] = await writer.ainvoke(state["messages"])

        return response

    async def reflection_node(state: ReflectionState) -> ReflectionState:
        # Other messages we need to adjust
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [state["messages"][0]] + [
            cls_map[msg.type](content=msg.content)
            for msg in state["messages"][1:]
        ]
        res = await reviewer.ainvoke(translated)
        # We treat the output of this as human feedback for the generator

        response = {}
        response["messages"] = [HumanMessage(content=res.content)]
        if "final" in state and state["final"]:
            response["previous"] = state["final"]
        response["final"] = state["messages"][-1].content
        return response

    def is_essay(state):
        topic = state["messages"][0].content  # TOTO -> prerobit na topic
        essay = state["messages"][-1].content
        decision = check_essay.invoke({"topic": topic, "essay": essay})

        if decision.startswith("Yes"):
            return "reflection"
        else:
            return END

    def check(state: ReflectionState) -> str:
        if "counter" not in state:
            state["counter"] = 1

        if state["counter"] > max_count:
            return END
        else:
            return "generate"

    builder = StateGraph(ReflectionState)
    builder.add_node("generate", generation_node)
    builder.add_node("reflection", reflection_node)
    builder.add_edge(START, "generate")
    # builder.add_edge("generate", "reflection")

    builder.add_conditional_edges("generate", is_essay)
    builder.add_conditional_edges("reflection", check)
    graph = builder.compile()

    return graph
