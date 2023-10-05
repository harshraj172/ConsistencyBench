import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from .prompt_template import QUESTION_TEMPLATE

class BaseGenerator():
    def __init__(self, model, vars_type):
        super(BaseGenerator, self).__init__()
        self.vars_type = vars_type
        self.model = model
        self.question_prompt = PromptTemplate(
                input_variables=["question"],
                template=QUESTION_TEMPLATE,)
    
    def apply(self, input, inputs_pert, outputs):
        return outputs
    
    def generate(self, input, input_perts):
        outputs = []
        if self.vars_type=="sampling":
            for temperature in np.arange(0, 2, 0.2):
                self.model.model_kwargs['temperature'] = temperature
                chain = LLMChain(llm=self.model, prompt=self.question_prompt)
                output = chain.run({"question":input,})
                outputs.append(output.strip())
        elif self.vars_type=="paraphrase":
            chain = LLMChain(llm=self.llm, prompt=self.question_prompt)
            input_perts = input + input_perts
            for input_pert in input_perts:
                output = chain.run({"question":input_pert,})
                outputs.append(output.strip())
        else:
            NotImplementedError
        outputs = self.apply(input, input_perts, outputs)
        return outputs