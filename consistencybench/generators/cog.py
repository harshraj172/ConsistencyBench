import re
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .prompt_template import QUESTION_TEMPLATE
from .base import BaseGenerator


class CoGGenerator(BaseGenerator):
    """
    Implements Chain-of-Guidance prompting, designed to create
    multiple-choice options based on input variations.

    Inherits from:
        BaseGenerator: The base class for generating output variations.
    """

    def __init__(self, model, variation_type):
        """
        Initializes the CoGGenerator with the given model and variation type.

        Args:
            model (Model): The language model to use.
            variation_type (str): The method of variation to apply ('sampling' or 'paraphrasing').
        """
        super(CoGGenerator, self).__init__(model, variation_type)
        self.variation_type = variation_type
        self.model = model
        self.question_prompt = PromptTemplate(
            input_variables=["question"],
            template=QUESTION_TEMPLATE,
        )

    def apply(self, input, input_perts, outputs):
        """
        Extends the apply method to create a multiple-choice format for each output.

        Args:
            input (str): The original input.
            input_perts (list): Perturbed versions of the input.
            outputs (list): Generated outputs to be transformed into multiple-choice format.

        Returns:
            list: Outputs formatted as multiple-choice questions.
        """
        from .prompt_template import CHOICE_TEMPLATE

        CHOICE_TEMPLATE_SUFFIX = ""
        for output in outputs:
            CHOICE_TEMPLATE_SUFFIX += f"""\n {output}"""
        CHOICE_TEMPLATE_SUFFIX += f"""\n Don't know the correct answer"""
        CHOICE_TEMPLATE_SUFFIX += """\n\nAnswer:"""
        CHOICE_TEMPLATE_SUFFIX = CHOICE_TEMPLATE_SUFFIX.replace("{", "{{")  # cleaning
        CHOICE_TEMPLATE_SUFFIX = CHOICE_TEMPLATE_SUFFIX.replace("}", "}}")  # cleaning
        CHOICE_TEMPLATE += CHOICE_TEMPLATE_SUFFIX
        choice_prompt = PromptTemplate(
            input_variables=[
                "question",
            ],
            template=CHOICE_TEMPLATE,
        )

        outputs = []
        if self.variation_type == "sampling":
            for temperature in np.arange(0, 2, 0.2):
                self.model.model_kwargs["temperature"] = temperature
                chain = LLMChain(llm=self.model, prompt=choice_prompt)
                output = chain.run(
                    {
                        "question": input,
                    }
                )
                outputs.append(output.strip())
        elif self.variation_type == "paraphrasing":
            chain = LLMChain(llm=self.model, prompt=choice_prompt)
            for input_pert in input_perts:
                output = chain.run(
                    {
                        "question": input_pert,
                    }
                )
                outputs.append(output.strip())
        else:
            NotImplementedError

        outputs = [
            re.sub(re.compile(r"Option [0-9]+"), "", str(s)).strip() for s in outputs
        ]  # cleaning
        outputs = [str(s).split(":")[-1].strip() for s in outputs]  # cleaning

        return outputs
