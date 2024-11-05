from typing import Annotated
from langchain_core.tools import tool
from patra_agent.util import graph, top_k_results

@tool
def execute_cypher(
        query: Annotated[str, "Cypher query to execute on the graph"],
) -> Annotated[str, "Result of the Cypher query execution"]:
    """Execute the given cypher query on the graph."""
    response = graph.query(query)[:top_k_results]
    query_result = str(response).replace("{", "{{").replace("}", "}}")
    print("\n\n+++++++++++++++++++++++")
    print("Running Cypher Query:")
    print(query_result)
    print("+++++++++++++++++++++++")
    print("+++++++++++++++++++++++\n\n")
    return query_result


@tool
def print_hello(
        name: Annotated[str, "Name of the person"],
) -> Annotated[str, "Hello + name"]:
    """Returns hello name."""
    return str("Hello, {}!".format(name))
