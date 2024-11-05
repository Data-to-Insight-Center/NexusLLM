from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from patra_agent.util import llm, graph

class QueryGenerator(BaseModel):
    """ Use this class to structure the output for the query"""
    cypher_query: str = Field(
        description="Syntactically correct cypher query ready for execution"
    )

    context: str = Field(
        description="Context about the query"
    )

query_agent_prompt = """You are an expert in writing Cypher queries for a Neo4j database. 
Only return the cypher query
Try and convert datetime when returning. 
Here's an example:
MATCH (u:User {{user_id: 'swithana'}})-[r:SUBMITTED_BY]-(e:Experiment)
RETURN e, datetime({{epochMillis: e.start_time}}) AS start_time

To get information about the ModelCard from the experiment use:
Match (exp:Experiment {{experiment_id: `d2i-exp-3442334`}})-[r:USED]-(m:Model)-[r2:USED]-(mc:ModelCard)
return mc

Models have a '-model' in their id. Do not drop it when querying the database. 

To get information about a modelCard from a model you can use:
MATCH (m:Model {{model_id: '33232113' }})-[r2:USED]-(mc:ModelCard) return mc

Only use the relationship [:VERSION_OF] to get similar ModelCards. 

You can compare test accuracy and other model attributes across model cards using this:
MATCH 
  (mc1:ModelCard {{external_id: 'example_1'}})-[r:USED]-(m1:Model), 
  (mc2:ModelCard {{external_id: 'example_2'}})-[r2:USED]-(m2:Model) 
WITH 
  m1, m2, 
  CASE 
    WHEN m1.test_accuracy > m2.test_accuracy THEN m1 
    ELSE m2 
  END AS highest_accuracy_model
RETURN highest_accuracy_model

PatternExpressions are not allowed to introduce new variables if the variable is not used anywhere.
For example, instead of:
MATCH (m:Model)
WHERE NOT (m)-[:USED]-(mc:ModelCard)
RETURN m

You should write:
MATCH (m:Model)
WHERE NOT (m)-[:USED]-(:ModelCard)
RETURN m

To get information about images executed in an experiement on a device, use this as
an example:

MATCH (img:RawImage)-[r2:PROCESSED_BY]-(e:Experiment)-[r:EXECUTED_ON]-(d:EdgeDevice) 
RETURN img

To get deployment information about model deployments you can use deployment node.
For deployment location, you can use the location field in EdgeDevice. 
Example: to find models deployed in raspi-3 device types in ohio-zoo location you can use:
match (m:Model)-[:HAS_DEPLOYMENT]-(depl:Deployment)-[:DEPLOYED_IN]-(device:EdgeDevice {{location:'ohio-zoo', device_type: 'raspi3'}})
 return m, depl.device_type, depl.average_accuracy, device.location, device.device_id

available model types for Model are [convolutional neural network, large language model, foundational model]. These are case sensitive.

external_id in certain nodes refer to the node ids. These must be returned with results. 

"""


query_kg_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_agent_prompt),
        ("human", "Only return the cypher query for the question: \n {question} \n Schema: {graph_schema}.",
        ),
    ]
)
query_agent_llm = llm.with_structured_output(QueryGenerator)
query_generator = query_kg_prompt | query_agent_llm