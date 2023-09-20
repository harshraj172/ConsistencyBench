import re
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from prompt_template import QUESTION_TEMPLATE
from base import BaseGenerator

class A2CGenerator(BaseGenerator):
    def __init__(self, model, vars_type):
        super(A2CGenerator, self).__init__()
        self.vars_type = vars_type
        self.model = model
        self.question_prompt = PromptTemplate(
                input_variables=["question"],
                template=QUESTION_TEMPLATE,)
        
    def apply(self, input, inputs_pert, outputs):

        CHOICE_TEMPLATE_SUFFIX = ""
        for output in outputs:
            CHOICE_TEMPLATE_SUFFIX += f"""\n {output}"""
        CHOICE_TEMPLATE_SUFFIX += f"""\n Don't know the correct answer"""
        CHOICE_TEMPLATE_SUFFIX += """\n\nAnswer:"""  
        CHOICE_TEMPLATE_SUFFIX = CHOICE_TEMPLATE_SUFFIX.replace('{', '{{') # cleaning 
        CHOICE_TEMPLATE_SUFFIX = CHOICE_TEMPLATE_SUFFIX.replace('}', '}}') # cleaning 
        CHOICE_TEMPLATE += CHOICE_TEMPLATE_SUFFIX
        choice_prompt = PromptTemplate(
            input_variables=["question",],
            template=CHOICE_TEMPLATE,)
        
        outputs = []
        if self.vars_type=="sampling":
            for temperature in np.arange(0, 2.5, 0.25):
                self.model.pipeline_kwargs["temperature"] = temperature
                chain = LLMChain(llm=self.model, prompt=choice_prompt)
                output = chain.run({"question":input,})
                outputs.append(output.strip())
        elif self.vars_type=="paraphrase":
            chain = LLMChain(llm=self.llm, prompt=choice_prompt)
            input_perts = input + input_perts
            for input_pert in input_perts:
                output = chain.run({"question":input_pert,})
                outputs.append(output.strip())
        else:
            NotImplementedError

        outputs = [re.sub(re.compile(r'Option [0-9]+'), '', str(s)).strip() for s in outputs] # cleaning
        outputs = [str(s).split(':')[-1].strip() for s in outputs] # cleaning

        return outputs