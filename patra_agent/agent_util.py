from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from patra_agent.util import llm


def create_agent(tools, template: str, system_message: str, llm=llm):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                template + "\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
            # MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)
