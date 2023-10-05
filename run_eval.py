import argparse
import numpy as np
import pandas as  pd
from tqdm.auto import tqdm

import torch
from datasets import load_dataset
from transformers import pipeline

from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
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
    parser.add_argument('--data_name',
                        default='truthful_qa',
                        type=str,
                        help="name of the data to benchmark model's consistency."
                        "Note: it should be available on huggingface dataset")
    parser.add_argument('--model_name',
                        type=str,
                        default="gpt-3.5-turbo")
    parser.add_argument('--aux_model_name',
                        type=str,
                        default="gpt-3.5-turbo")
    parser.add_argument('--openai_api_key',
                        type=str,)
    parser.add_argument('--perturb_type',
                        type=str,
                        choices=["paraphrasing", "sampling"],
                        default="sampling",)
    parser.add_argument('--scoring_type',
                        type=str,
                        choices=["entropy", "pairwise"],
                        default="pairwise",)
    parser.add_argument('--variation_type',
                        type=str,
                        choices=["paraphrasing", "sampling"],
                        default="sampling",)
    parser.add_argument('--eval_agreements',
                        default='llm,0.5;contradiction,0.5;ner,0.5',
                        type=str,
                        help="output consistency evaluation strategy to use",)
    args = parser.parse_args()

    if args.data_name=='truthful_qa':
        data = load_dataset('truthful_qa', 'generation')
        df = data['validation'].to_pandas()
    else:
        NotImplementedError
    agreements = [tuple(x.split(',')) for x in args.eval_agreements.split(';')]
    
    if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
        model = ChatOpenAI(model_name=args.model_name, openai_api_key=args.openai_api_key, top_p=0.5)
    else:
        if 't5' in args.model_name:
            model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text2text-generation")
        else:
            model = HuggingFacePipeline.from_model_id(model_id=args.model_name, task="text-generation")

    aux_model = None
    if 'llm' in [x for x, _ in agreements]:
        if args.aux_model_name in ["gpt-3.5-turbo", "gpt-4"]:
            aux_model = ChatOpenAI(model_name=args.aux_model_name, openai_api_key=args.openai_api_key, top_p=0.0)
        else:
            if 't5' in args.aux_model_name:
                pipe = pipeline(model=args.aux_model_name, task="text2text-generation", temperature=0.1, device_map="auto")
                aux_model = HuggingFacePipeline(pipeline=pipe)
            else:
                pipe = pipeline(model=args.aux_model_name, task="text-generation", temperature=0.1, device_map="auto")
                aux_model = HuggingFacePipeline(pipeline=pipe)

    a2c = A2CGenerator(model, args.variation_type)
    base = BaseGenerator(model, args.variation_type)
    scorer = ConsistencyScorer(agreements, args.scoring_type, aux_model)
    all_input, all_input_perturb, all_output, all_output_cons, all_correct_output, all_scores, all_cons_scores = [], [], [], [], [], [], []
    for i in tqdm(range(len(df))):
        input = df.question[i]
        correct_output = df.best_answer[i]
        
        if args.perturb_type=="paraphrasing":
            input_perts = [paraphrase.llm_prompting(input, idx) for idx in range(1, len(5))]
        else:
            input_perts = []
    
        # Generating Outputs
        outputs = base.generate(input, input_perts)
        cons_outputs = a2c.generate(input, input_perts)

        ## Scoring Outputs
        score = scorer.score(input, outputs)
        cons_score = scorer.score(input, cons_outputs)

        all_input.extend([input]*len(outputs))
        all_input_perturb.extend([input]+input_perts) if input_perts else all_input_perturb.extend(['']*len(outputs))
        all_output.extend(outputs)
        all_output_cons.extend(cons_outputs)
        all_correct_output.extend([correct_output]*len(outputs))
        all_scores.extend([score]*len(outputs))
        all_cons_scores.extend([cons_score]*len(outputs))
        
        res_df = pd.DataFrame({
            "input": all_input,
            "input_pert": all_input_perturb,
            "outputs_correct": all_correct_output,
            "output_sampled": all_output,
            "output_consistent": all_output_cons,
            "score": all_scores,
            "score_consistent": all_cons_scores,
        })
        
        res_df.to_csv(f"result_{args.model_name.replace('/', '')}_{args.perturb_type}.csv", index=False)