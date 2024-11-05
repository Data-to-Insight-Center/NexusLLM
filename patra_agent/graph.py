from pprint import pprint
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from patra_agent.graph_state import PatraState
from patra_agent.query_agent import query_generator
from patra_agent.util import graph, top_k_results
from patra_agent.db_agent import db_executor
from patra_agent.patra_agent import patra_executor
from typing import Literal


PATRA_AGENT_NAME = "patra_agent"
QUERY_AGENT_NAME = "query_agent"
DB_AGENT_NAME = "db_executor"

def patra_node(state: PatraState) -> PatraState:
    response = patra_executor.invoke({"messages": state.messages})
    result = AIMessage(**response.dict(exclude={"type", "name"}), name=PATRA_AGENT_NAME)
    state.messages.append(result)
    state.sender = PATRA_AGENT_NAME
    return state

def cypher_generator_node(state: PatraState) -> PatraState:
    query_question = state.messages[-1].content
    cypher_query = query_generator.invoke({"graph_schema": state.graph_schema, "question": query_question}).cypher_query
    state.messages.append(AIMessage(content=str(cypher_query), name=QUERY_AGENT_NAME))
    return state

def execute_query_node(state: PatraState) -> PatraState:
    query = state.messages[-1].content
    question = state.messages[-2].content
    response = db_executor.invoke({"query": query, "original_question": question})
    state.messages.append(AIMessage(content=response['output'], name=DB_AGENT_NAME))
    return state

def job_agent(state: PatraState) -> PatraState:
    # Use QueryAgent and submit job
    return {"messages": [AIMessage(content="Job submitted")]}

def research_agent(state: PatraState) -> PatraState:
    # Use GoogleScholar API and QueryAgent
    return {"messages": [AIMessage(content="Research completed")]}

def supervisor(state: PatraState) -> str:
    # Decide which agent to invoke based on the last message
    last_message = state["messages"][-1].content
    if "query" in last_message.lower():
        return "query_agent"
    elif "job" in last_message.lower():
        return "job_agent"
    elif "research" in last_message.lower():
        return "research_agent"
    else:
        return END

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state.messages
    last_message = messages[-1]
    # if isinstance(last_message, ToolMessage) and last_message.tool_calls:
    #     # The previous agent is invoking a tool
    #     return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

# Create the graph
patra = StateGraph(PatraState)
patra.graph_schema = str(graph.get_structured_schema)

# Add nodes
patra.add_node(PATRA_AGENT_NAME, patra_node)
patra.add_node(DB_AGENT_NAME, execute_query_node)
patra.add_node(QUERY_AGENT_NAME, cypher_generator_node)

# Set entry point
patra.set_entry_point(PATRA_AGENT_NAME)

# QueryAgent can interact with DBAgent multiple times
patra.add_edge(QUERY_AGENT_NAME, DB_AGENT_NAME)
patra.add_edge(DB_AGENT_NAME, PATRA_AGENT_NAME)

patra.add_conditional_edges(
    PATRA_AGENT_NAME,
    router,
    {"continue": QUERY_AGENT_NAME, "__end__": END},
)

# Compile the graph
app = patra.compile()


def run_patra_graph(question):
    # result = app.invoke({
    #     "messages": [HumanMessage(content=question)],
    #     "sender": "human",
    #     "graph_schema": str(graph.get_structured_schema),
    # })
    inputs = {
        "messages": [HumanMessage(content=question)],
        "sender": "human"
    }
    for chunk in app.stream(inputs, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    return ""
