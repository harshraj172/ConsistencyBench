import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .prompt_template import QUESTION_TEMPLATE


class BaseGenerator:
    """
    Base class for generating variations of outputs based on different methods.

    Attributes:
        variation_type (str): Type of variation method used (e.g., 'sampling', 'paraphrasing').
        model (Model): The language model used for generating outputs.
        question_prompt (PromptTemplate): Template for formatting the input question.
    """

    def __init__(self, model, variation_type):
        """
        Initializes the BaseGenerator with a specified model and variation type.

        Args:
            model (Model): The language model to use.
            variation_type (str): The method of variation to apply ('sampling' or 'paraphrasing').
        """
        super(BaseGenerator, self).__init__()
        self.variation_type = variation_type
        self.model = model
        self.question_prompt = PromptTemplate(
            input_variables=["question"],
            template=QUESTION_TEMPLATE,
        )

    def apply(self, input, inputs_pert, outputs):
        """
        Applies additional processing to the generated outputs if needed.

        Args:
            input (str): The original input.
            inputs_pert (list): Perturbed versions of the input.
            outputs (list): Generated outputs before final processing.

        Returns:
            list: Processed outputs.
        """
        return outputs

    def generate(self, input, input_perts):
        """
        Generates variations of outputs based on the specified variation type.

        Args:
            input (str): The original input to generate variations from.
            input_perts (list): A list of perturbed inputs for variation generation.

        Returns:
            list: A list of generated output variations.
        """
        outputs = []
        if self.variation_type == "sampling":
            for temperature in np.arange(0, 2, 0.2):
                self.model.model_kwargs["temperature"] = temperature
                chain = LLMChain(llm=self.model, prompt=self.question_prompt)
                output = chain.run(
                    {
                        "question": input,
                    }
                )
                outputs.append(output.strip())
        elif self.variation_type == "paraphrasing":
            chain = LLMChain(llm=self.model, prompt=self.question_prompt)
            input_perts = [input] + input_perts
            for input_pert in input_perts:
                output = chain.run(
                    {
                        "question": input_pert,
                    }
                )
                outputs.append(output.strip())
        else:
            NotImplementedError
        outputs = self.apply(input, input_perts, outputs)
        return outputs
