from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from .prompt_template import * 

def llm_prompting(input, method=1):
    pp_prompt = PromptTemplate(
            input_variables=["method", "sentence"],
            template=PP_TEMPLATE,
        )
    llm = ChatOpenAI(openai_api_key="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",model_name="gpt-3.5-turbo")
    messages = [HumanMessage(content=pp_prompt.format(method=str(method), sentence=input))]
    input_pp = llm(messages, stop='\n').content
    return input_pp.strip()