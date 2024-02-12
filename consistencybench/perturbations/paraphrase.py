from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from .prompt_template import PP_TEMPLATE


def llm_prompting(input, method=1):
    """
    Generate a paraphrase from input text using a language model (LLM) based on a specified template.

    Args:
        input_text (str): The text to be processed.
        method (int, optional): The method number to be used for processing.

    Returns:
        str: The processed text output from the language model.
    """
    pp_prompt = PromptTemplate(
        input_variables=["method", "sentence"],
        template=PP_TEMPLATE,
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    messages = [
        HumanMessage(content=pp_prompt.format(method=str(method), sentence=input))
    ]
    input_pp = llm(messages, stop="\n").content
    return input_pp.strip()
