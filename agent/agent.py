from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.chains.llm import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import os 
import streamlit as st 
from decouple import config
from sqlalchemy.engine.base import Connection, Engine
from .custom_tools import QuerySaveSQLDataBaseTool
from .prompts import sql_helper_prompt_template, sql_db_query_description, sql_db_schema_description
from .custom_parsers import CustomPromptTemplate, CustomOutputParser
from .custom_tools import CustomSQLDatabase


os.environ["OPENAI_API_TYPE"] = config("OPENAI_API_TYPE")
os.environ["OPENAI_API_BASE"] = config("OPENAI_API_BASE")
os.environ["OPENAI_API_VERSION"] = config("OPENAI_API_VERSION")
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


@st.cache_resource(hash_funcs={Engine:id})
def connect_with_langchain_db(engine):
    db = CustomSQLDatabase(engine, view_support=True)
    return db


def get_sql_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm, temperature=0)
    query_and_save_tool = QuerySaveSQLDataBaseTool(db=db)
    tools = toolkit.get_tools()
    tools.append(query_and_save_tool)
    tools[0].description = sql_db_query_description
    tools[1].description = sql_db_schema_description
    tool_names = [tool.name for tool in tools]

    return tools, tool_names


def get_agent(snowflake_db):
    #creating llm 
    llm_chat_model = AzureChatOpenAI(deployment_name="gpt-4-32k", model_name="gpt-4-32k", temperature=0)

    tools, tool_names = get_sql_tools(db=snowflake_db, llm=llm_chat_model)
    prompt = CustomPromptTemplate(
        template=sql_helper_prompt_template,
        tools=tools,
        query_and_save_tool=tools[-1].name,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history", "message_id"]
    )

    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=llm_chat_model, prompt=prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    message_history = StreamlitChatMessageHistory()

    memory = ConversationTokenBufferMemory(
        memory_key="history",
        chat_memory=message_history,
        llm=llm_chat_model,
        max_token=6000, 
        output_key="output", 
        input_key="input",
        return_messages=True
    )


    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory,
        return_intermediate_steps=True
    )
    
    return agent_executor




def clean_chat_memory():
    # message_history = StreamlitChatMessageHistory()
    # message_history.clear()
    st.session_state.chat_memory = []
