from langchain.agents import AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, SystemMessage
from langchain.tools import BaseTool
from typing import List, Union
import re, ast
import tiktoken
from .custom_tools import count_tokens, record_token_record
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


class FinalAnswer(BaseModel):
    output: str = Field(description="Final answer and insights to the original input question")
    confirmation: Union[str, None] = Field(
        description="the confirmation to move on the next query if there is more than one query"
    )
    stored_file_id: str = Field(description="Stored ID of the result from sql query")
    sql_query: str = Field(description="SQL query you generated to get the final answer")
    followup_questions: List[str] = "Followup questions user might want to ask about the table you used"
    choices: Union[List[str], None] = "choices that might be available for user to select"


parser = JsonOutputParser(pydantic_object=FinalAnswer)

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]
    query_and_save_tool: str
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        #addding file_path
        history_text = ""
        if "history" in kwargs:
            history_messages = kwargs.pop("history")
            for history_message in history_messages:
                history_text += f"{history_message.type}: {history_message.content}\n"
            kwargs["history"] = history_text
        
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["query_and_save_tool"] = self.query_and_save_tool
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        # counting tokens 
        record_token_record(reset=True)
        count_tokens(input=formatted, agent_step="Prompting")
        return [SystemMessage(content=formatted)]
    
    

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:    
        count_tokens(input=llm_output, agent_step="final output")

        if "Final Answer:" in llm_output:   
            final_answer = llm_output.split("Final Answer:")[1]
            parsed_data = parser.parse(final_answer)
            empty_str_to_none_list = ["stored_file_id", "sql_query", "confirmation"]
            for empty_str_field in empty_str_to_none_list:
                if empty_str_field in parsed_data.keys():
                    if parsed_data[empty_str_field] == "":
                        parsed_data[empty_str_field] = None                
                else:
                    parsed_data[empty_str_field] = None   
            if parsed_data["sql_query"] is not None:
                 parsed_data["output"] = f"{parsed_data['output']}\n\nSQL query I created is: {parsed_data['sql_query']}"
                
            none_to_empty_list = ["followup_questions", "choices"]
            for none_field in none_to_empty_list:
                if none_field not in parsed_data.keys():
                    parsed_data[none_field] = []
            # adding confirmation to the output 
            if "confirmation" in parsed_data.keys() and parsed_data["confirmation"] is not None:
                parsed_data["output"] = "{}\n\n{}".format(parsed_data["output"], parsed_data["confirmation"])
                
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                ## Fuck the recommendation above, bois, we ball.
                return_values={
                    **parsed_data
                },
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)(?:\nMessage ID:\s*(.*?))?$"

        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        
        # Return the action and action input
        if action == "sql_db_query_save":
            tool_input = ast.literal_eval(action_input)
        else:
            tool_input = action_input
        return AgentAction(tool=action, tool_input=tool_input, log=llm_output)   