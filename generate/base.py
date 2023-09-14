import numpy as np
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from prompt_template import QUESTION_TEMPLATE

class BaseGenerator():
    def __init__(self, model, vars_type):
        super(BaseGenerator, self).__init__()
        self.vars_type = vars_type
        self.model = model
        self.question_prompt = PromptTemplate(
                input_variables=["question"],
                template=QUESTION_TEMPLATE,)
    
    def apply(input, inputs_pert, outputs):
        return outputs
    
    def generate(self, input, inputs_pert):
        outputs = []
        if self.vars_type=="sampling":
            for temperature in np.arange(0, 2.5, 0.25):
                self.model.pipeline_kwargs["temperature"] = temperature
                chain = LLMChain(llm=self.model, prompt=self.question_prompt)
                output = chain.run({"question":input,})
                outputs.append(output.strip())
        elif self.vars_type=="paraphrase":
            chain = LLMChain(llm=self.llm, prompt=self.question_prompt)
            for input_pert in inputs_pert:
                output = chain.run({"question":input_pert,})
                outputs.append(output.strip())
        else:
            NotImplementedError
        outputs = self.apply(input, inputs_pert, outputs)
        return outputs