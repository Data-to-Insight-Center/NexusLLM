from typing import Annotated, Sequence, TypedDict, List, ClassVar
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
from pydantic import BaseModel, Field
from patra_agent.util import graph

class PatraState(BaseModel):
    """
    Agent state for the Agentic workflow.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # graph schema for the CKN graph
    graph_schema: ClassVar = str(graph.get_structured_schema)
    # sender of the last message
    sender: str = Field(default="human")
