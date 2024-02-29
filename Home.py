import streamlit as st
from agent import get_agent, connect_with_langchain_db, clean_chat_memory
from agent.custom_tools import record_token_usage, record_token_record, get_token_usage
from helpers import get_engine, get_intermediate_steps_str, get_output_parts
from pandas.core.frame import DataFrame 
import random, string, datetime



st.title("☃️ SQL Agent")
st.button("Clear history", on_click=clean_chat_memory)
show_followup_questions_toggle = st.toggle('Show followup questions')
show_immediate_steps_toggle = st.toggle('Show immediate steps')

engine = get_engine()
connection = engine.connect()

with st.spinner("Getting tables..."):
    connection_db = connect_with_langchain_db(engine)

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

agent_executor = get_agent(connection_db)
for message in st.session_state.chat_memory:
    with st.chat_message(message["type"]):
        if message["type"] == "human":
            st.write(message["content"])
        elif message["type"] == "AI":
            final_answer, dataframe, sql_query, followup_questions = get_output_parts(response=message, message_id=message["message_id"])
            st.write(final_answer)
            if type(dataframe) == DataFrame:
                st.write("DataFrame:")
                st.dataframe(dataframe)
            if sql_query:
                st.write(f"SQL Query:\n{sql_query}")                
            if show_followup_questions_toggle and followup_questions:
                st.write(f"Followup Questions:")
                for num, followup_question in enumerate(followup_questions):
                    st.write(f"{num+1}. {followup_question}")
            st.write(f"Agent time: {message['agent_time']}")
                
            if show_immediate_steps_toggle and "intermediate_steps_string" in message:
                st.write("MY INTERMEDIATE STEPS: \n\n", message["intermediate_steps_string"])
                print(message["intermediate_steps_string"])

chat_input = st.chat_input("Chat here")
if chat_input:
    with st.chat_message('human'):
        st.write(chat_input) 
        # storing the data
        st.session_state.chat_memory.append(
            {
                "type" : "human",
                "content": chat_input
            }
        )
    
    with st.spinner("Thinking..."):
        #random message id 
        message_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        # invoking the agent
        time_start = datetime.datetime.now()
        response = agent_executor.invoke({"input": chat_input, "message_id": message_id})
        time_finish = datetime.datetime.now()
        agent_time = time_finish - time_start
        agent_time_min = int(agent_time.total_seconds() // 60)
        agent_time_seconds = int(agent_time.total_seconds() % 60)
        agent_time_str = f"{agent_time_min} min, {agent_time_seconds} seconds"
        final_answer, dataframe, sql_query, followup_questions = get_output_parts(response=response, message_id=message_id)
        intermediate_steps_string = get_intermediate_steps_str(response)
        # reading token usage 
        used_tokens = get_token_usage()
        # adding last action log 
        with st.chat_message('ai'):
            st.write(final_answer)
            if type(dataframe) == DataFrame:
                st.write("DataFrame:")
                st.dataframe(dataframe)
            if sql_query:
                st.write(f"SQL Query:\n{sql_query}")      
            if show_followup_questions_toggle and followup_questions:
                st.write(f"Followup Questions:")
                for num, followup_question in enumerate(followup_questions):
                    st.write(f"{num+1}. {followup_question}")
            st.write(f"Agent time: {agent_time_str}")
            if show_immediate_steps_toggle:
                st.write("MY INTERMEDIATE STEPS: \n\n", intermediate_steps_string)

            # storing the data
            ai_response = {
                "type" : "AI",
                "intermediate_steps_string":intermediate_steps_string,
                "output": final_answer,
                "message_id": message_id,
                "stored_file_id": response["stored_file_id"],
                "sql_query": sql_query,
                "followup_questions": followup_questions,
                "used_tokens": used_tokens,
                "agent_time": agent_time_str
            }
            st.session_state.chat_memory.append(ai_response)
        # resetting token records
        record_token_record(reset=True)
        record_token_usage(reset=True)

                
