from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from patra_agent.util import llm, graph
from patra_agent.tools import execute_cypher

tools = [execute_cypher]

answer_generator_template = """You are tasked with executing a given query in a neo4j graph and returning the response.
You have access to the execute_cypher tool which lets you execute the given query on the neo4j databse. 
  
Context for the query is the following question from a user: {original_question}  

Keep the output structured if possible.
 Don't use bold, underline or other text altering stuff. Just plain text. 
 Do not return the embeddings specially in the ModelCard information. 
 for start time and end times convert to human readable timestamps. These are EDT. """


db_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_generator_template),
        ("human", "Return the result to the query: \n {query}."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
db_agent_w_tools = create_tool_calling_agent(llm, tools, db_agent_prompt)
db_executor = AgentExecutor(agent=db_agent_w_tools, tools=tools, verbose=True)
