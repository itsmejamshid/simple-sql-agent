import tiktoken
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from .custom_tools import QuerySaveSQLDataBaseTool
from .prompts import sql_db_query_description, sql_db_schema_description


def get_sql_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm, temperature=0)
    query_and_save_tool = QuerySaveSQLDataBaseTool(db=db)
    tools = toolkit.get_tools()
    tools.append(query_and_save_tool)
    tools[0].description = sql_db_query_description
    tools[1].description = sql_db_schema_description
    tool_names = [tool.name for tool in tools]

    return tools, tool_names
