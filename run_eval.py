import argparse
import numpy as np
import pandas as  pd
from tqdm.auto import tqdm

import torch
from transformers import pipeline

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from generate import A2CGenerator, BaseGenerator
from evaluate import ConsistencyScorer
from perturb import paraphrase

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--input_file',
                        type=str,
                        help='path to data in .csv')
    parser.add_argument('--model_name',
                        type=str,
                        default="text-davinci-003")
    parser.add_argument('--aux_model_name',
                        type=str,
                        default="text-davinci-003")
    parser.add_argument('--openai_api_key',
                        type=str,
                        default="text-davinci-003")
    parser.add_argument('--perturb_type',
                        type=str,
                        choices=["paraphrasing", "sampling"],
                        default="sampling",)
    parser.add_argument('--variation_type',
                        type=str,
                        choices=["paraphrasing", "sampling"],
                        default="sampling",)
    parser.add_argument('--gneration_strategy',
                        type=str,
                        choices=["a2c", "base"],
                        default="output generation strategy to use",)
    parser.add_argument('--eval_agreements',
                        type=str,
                        default="output consistency evaluation strategy to use",)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    agreements = args.eval_agreements.split(',')
    
    if args.model_name in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]:
        model = OpenAI(openai_api_key=args.openai_api_key, top_p=0.7)
    else:
        if 't5' in args.model_name:
            model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text2text-generation")
        else:
            model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text-generation")

    if args.eval_strategy=="llm_prompting":
        if args.aux_model_name in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]:
            aux_model = OpenAI(openai_api_key=args.openai_api_key, top_p=0.3)
        else:
            if 't5' in args.model_name:
                aux_model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text2text-generation")
            else:
                aux_model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text-generation")

    a2c = A2CGenerator(model, args.vars_type)
    base = BaseGenerator(model, args.vars_type)
    scorer = ConsistencyScorer(agreements)
    all_input, all_output, all_output_cons, all_correct_output, all_scores, all_cons_scores = [], [], [], [], [], []
    for i in tqdm(range(len(df))):
        input = df.question[i]
        correct_output = df.best_answer[i]
        
        if args.perturb_type=="paraphrasing":
            input_pert = [paraphrase.llm_prompting(input, idx) for idx in range(1, len(5))]
        else:
            input_pert = []
            
        # Generating Outputs
        outputs = base.generate(input, input_pert, vars_type=args.perturb_type)
        cons_outputs = a2c.generate(input, input_pert, vars_type=args.perturb_type)

        ## Scoring Outputs
        score = scorer.score(input, outputs)
        cons_score = scorer.score(input, cons_outputs)

        all_input.extend([input, input_pert])
        all_output.extend(outputs)
        all_output_cons.extend(cons_outputs)
        all_correct_output.extend([correct_output]*len(input_pert))

        all_scores.extend([score]*len(outputs))
        all_cons_scores.extend([cons_score]*len(outputs))

        res_df = pd.DataFrame({
            "input": all_input,
            "outputs_correct": all_correct_output,
            "output_sampled": all_output,
            "output_consistent": all_output_cons,
            "score": all_scores,
            "score_consistent": all_cons_scores,
        })
        
        res_df.to_csv(f"result_{args.model_name.replace('/', '')}_{args.perturb_type}", index=False)