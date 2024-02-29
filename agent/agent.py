from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.chains.llm import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from .prompts import retriever_tool_description
import os 
import streamlit as st 
from decouple import config
from sqlalchemy.engine.base import Connection, Engine
from .custom_tools import QuerySaveSQLDataBaseTool
from .prompts import sql_helper_prompt_template, sql_db_query_description
from .custom_parsers import CustomPromptTemplate, CustomOutputParser
from .custom_tools import CustomSQLDatabase


os.environ["OPENAI_API_TYPE"] = config("OPENAI_API_TYPE")
os.environ["AZURE_OPENAI_ENDPOINT"] = config("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = config("OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_API_KEY"] = config("AZURE_OPENAI_API_KEY")


llm_embedding_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002", 
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)

thread_vector_db = FAISS.load_local("agent/faiss/thread_data", embeddings=llm_embedding_model)
cyber_vector_db = FAISS.load_local("agent/faiss/cybersyn_data", embeddings=llm_embedding_model)


thread_retriever = thread_vector_db.as_retriever()
cyber_retriever = cyber_vector_db.as_retriever()


def get_retriever_tool(db_type):
    if db_type == "thread":
        return create_retriever_tool(
            thread_retriever,
            "search_necessary_table_schemas",
            retriever_tool_description
        )  
    return create_retriever_tool(
        cyber_retriever,
        "search_necessary_table_schemas",
        retriever_tool_description
    ) 


@st.cache_resource(hash_funcs={Engine:id})
def connect_with_langchain_db(engine):
    db = CustomSQLDatabase(engine, view_support=True)
    return db


def get_sql_tools(db, db_type):
    toolkit = SQLDatabaseToolkit(db=db, llm=AzureChatOpenAI(deployment_name='gpt-4-turbo', temperature=0))
    tools = toolkit.get_tools()
    query_and_save_tool = QuerySaveSQLDataBaseTool(db=db)
    retriever_tool = get_retriever_tool(db_type)
    tools.append(query_and_save_tool)
    tools.append(retriever_tool)
    tools[0].description = sql_db_query_description

    tools = tools[3:]
    tool_names = [tool.name for tool in tools]
    print(tool_names, " tool_names")
    return tools, tool_names


def get_agent(snowflake_db):
    #creating llm 
    llm_chat_model = AzureChatOpenAI(deployment_name="gpt-4-turbo", model_name="gpt-4-turbo", temperature=0)

    tools, tool_names = get_sql_tools(db=snowflake_db, db_type="thread")
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
        return_messages=True,
    )


    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True
        
        
    )
    
    return agent_executor




def clean_chat_memory():
    message_history = StreamlitChatMessageHistory()
    message_history.clear()
    st.session_state.chat_memory = []
