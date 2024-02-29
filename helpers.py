from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from decouple import config
import streamlit as st
import pandas as pd
import re, os


@st.cache_resource
def get_engine():
    url = URL(
        user=config("sn_username"),
        password=config("sn_password"),
        account=config("sn_account"),
        warehouse=config("sn_warehouse"),
        database=config("sn_database"),
        schema=config("sn_schema"),
        role=config("sn_role"),

    )
    engine = create_engine(url)
    return engine

@st.cache_data
def get_df_file(output_table_name):
    url = URL(
        user=config("sn_username"),
        password=config("sn_password"),
        account=config("sn_account"),
        warehouse=config("sn_warehouse"),
        database=config("sn_database"),
        schema=config("sn_schema"),
        role=config("sn_role"),

    )
    engine = create_engine(url)
    connection = engine.connect()
    sql_query = f"SELECT * FROM PROD_USE_CASES.ALAMO.{output_table_name.upper()}"
    df = pd.read_sql(sql_query, connection)
    return df.to_csv().encode('utf-8')


def get_intermediate_steps_str(response):
    final_str = ''
    for step in response["intermediate_steps"]:
        agent_action, observation = step
        if observation.startswith("DATA: "):
            observation = "DataFrame below"
        action_log = agent_action.log

        final_str += f"{action_log}\n\nObservation: {observation}\n\n"
    if final_str == '':
        final_str = "*Do not have intermediate steps*"
    return final_str


def get_output_parts(response, message_id):
    final_answer = response["output"] 
    followup_questions = response["followup_questions"]
    sql_query = response["sql_query"]
    
    if sql_query is not None and not sql_query.startswith("```"):
        sql_query = f"```sql\n{sql_query}\n```"

    dataframe = None
    if response['stored_file_id']:
        last_stored_file = f"{response['stored_file_id']}.csv"
        # deleting other tables
        message_folder_path = os.path.join("stored_data", message_id)
        if os.path.exists(message_folder_path):
            all_file_names = os.listdir(message_folder_path)
            for file_name in all_file_names:
                if file_name != last_stored_file:
                    # removing files
                    os.remove(os.path.join("stored_data", message_id, file_name))
            # reading last stored_data
            file_path = os.path.join(message_folder_path, last_stored_file)
            dataframe = pd.read_csv(file_path)
    
    return final_answer, dataframe, sql_query, followup_questions