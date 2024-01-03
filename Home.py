import streamlit as st
from agent.agent import get_agent, connect_with_langchain_db, clean_chat_memory
from agent.custom_tools import record_token_usage, record_token_record, get_token_usage
from helpers import get_engine, get_intermediate_steps_str, get_output_parts
from pandas.core.frame import DataFrame 
import random, string



st.title("☃️ SQL Agent")
st.button("Clear history", on_click=clean_chat_memory)
show_immediate_steps_toggle = st.toggle('Show immediate steps')
show_token_usage = st.toggle("Show token usage")

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
            final_answer, dataframe, followup_questions = get_output_parts(response=message, message_id=message["message_id"])
            st.write(final_answer)
            if type(dataframe) == DataFrame:
                st.write("DataFrame:")
                st.dataframe(dataframe)
            if followup_questions:
                st.write(f"Followup Questions:\n{followup_questions}")
            if show_token_usage:
                st.write(f"Used token amount: {message['used_tokens']}")
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
        response = agent_executor.invoke({"input": chat_input, "message_id": message_id})
        final_answer, dataframe, followup_questions = get_output_parts(response=response, message_id=message_id)
        intermediate_steps_string = get_intermediate_steps_str(response)
        # reading token usage 
        used_tokens = get_token_usage()
        # adding last action log 
        with st.chat_message('ai'):
            st.write(final_answer)
            if type(dataframe) == DataFrame:
                st.write("DataFrame:")
                st.dataframe(dataframe)
            if followup_questions:
                st.write(f"Followup Questions:\n{followup_questions}")
            if show_token_usage:
                st.write(f"Used token amount: {used_tokens}")
            if show_immediate_steps_toggle:
                st.write("MY INTERMEDIATE STEPS: \n\n", intermediate_steps_string)

            # storing the data
            ai_response = {
                "type" : "AI",
                "intermediate_steps_string":intermediate_steps_string,
                "output": final_answer,
                "message_id": message_id,
                "stored_id": response["stored_id"],
                "followup_questions": followup_questions,
                "used_tokens": used_tokens
            }
            st.session_state.chat_memory.append(ai_response)
        # resetting token records
        record_token_record(reset=True)
        record_token_usage(reset=True)

                
