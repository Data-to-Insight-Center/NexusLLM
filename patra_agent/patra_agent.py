from patra_agent.agent_util import create_agent
from patra_agent.tools import print_hello
from patra_agent.util import llm, graph, llama31

tools = [print_hello]
patra_agent_template = """You are an helpful AI assistant that's helping to understand the data in a database.
                You do not run any tools. 
                If you are unable to fully answer, that's OK, another assistant will help.  
                 will help where you left off. Execute what you can to make progress.
                                
                The database is a graph database containing models, modelcards, experiements, images, users ...etc. Do not ask the user anything.  

                You are really good at figuring out which questions to ask in which order to answer the original question. 
                Looking at the history, ask questions get the answers and solve the original question.
                
                You are only communicating with another agent who will get you information from the database. So you need to be precise with your questions. 

                The model id contains '-model' do not drop that when querying. 
                If the returned tool result is empty, rephrase the question and try again.
                
                If you are satisfied with the response to the original user query, add prefix FINAL ANSWER in the response along with the completed answer.
                 """

patra_executor = create_agent(
    [print_hello],
    template=patra_agent_template,
    system_message="Return a single question in english",
    llm=llm
)