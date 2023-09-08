import random
import argparse
import numpy as np
import pandas as  pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from templates import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def produce_output_variations(inp, type_="sampling"): 
    PROMPT_TEMPLATE = \
"""
Question: {question}
Answer the above question in a single sentence.
Answer:"""
# """
# Question: {question}
# Answer the above question in the fewest words possible.
# Answer:"""
    prompt = PromptTemplate(
            input_variables=["question"],
            template=PROMPT_TEMPLATE,)
    
    outs, inp_pps = [], []
    if type_ == "sampling":
        for t in np.arange(0.01, 1, 0.1):
            if args.model_name=="text-davinci-003":
                llm = OpenAI(openai_api_key="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", top_p=0.7, temperature=t)
                chain = LLMChain(llm=llm, prompt=prompt)
                out = chain.run({"question":inp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=t,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp), '')
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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", top_p=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            inp_pps.append(inp_pp)
            if args.model_name=="text-davinci-003":
                out = chain.run({"question":inp_pp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp_pp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=0.6,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp_pp), '')
            outs.append(out.strip())
    return outs, inp_pps

def ans_via_comparison(inp, outs, type_="sampling"):
#     # for all others
#     PROMPT_TEMPLATE = \
# """
# Question: {question}
# For the question above there are several options given below, choose one among them which seems to be the most correct."""
#     PROMPT_TEMPLATE_SUFFIX = ""
#     for i in range(len(outs)):
#         PROMPT_TEMPLATE_SUFFIX += f"""\nOption {i+1}: {outs[i]}"""
#     PROMPT_TEMPLATE_SUFFIX += f"""\nOption {len(outs)+1}: Don't know the correct answer"""
#     PROMPT_TEMPLATE_SUFFIX += """\n\nAnswer:"""  
#     PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('{', '{{')
#     PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('}', '}}')
    
    # for flan-T5-xl
    PROMPT_TEMPLATE = \
"""
Question: {question}
Instruction: Choose the correct option."""
    PROMPT_TEMPLATE_SUFFIX = ""
    for i in range(len(outs)):
        PROMPT_TEMPLATE_SUFFIX += f"""\n {outs[i]}"""
    PROMPT_TEMPLATE_SUFFIX += f"""\n Don't know the correct answer"""
    PROMPT_TEMPLATE_SUFFIX += """\n\nAnswer:"""  
    PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('{', '{{')
    PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('}', '}}')
    PROMPT_TEMPLATE += PROMPT_TEMPLATE_SUFFIX
    prompt = PromptTemplate(
        input_variables=["question",],
        template=PROMPT_TEMPLATE,)
    
    outs = []
    if type_ == "sampling":
        for t in np.arange(0.01, 1, 0.1):
            if args.model_name=="text-davinci-003":
                llm = OpenAI(openai_api_key="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", top_p=0.7, temperature=t)
                chain = LLMChain(llm=llm, prompt=prompt)
                out = chain.run({"question":inp})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=t,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp), '')
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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", top_p=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            if args.model_name=="text-davinci-003":
                out = chain.run({"question":inp_pp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp_pp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=0.6,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp_pp), '')
            outs.append(out.strip())
    return outs




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
    parser.add_argument('--perturb_type',
                        type=str,
                        choices=["paraphrasing", "sampling"],
                        default="sampling",)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    
    if args.model_name not in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]:
        if "t5" in args.model_name:
            tokenizer = T5Tokenizer.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')

    
    all_input, all_output, all_input_pert, all_correct_output = [], [], [], [], [], []
    for i in tqdm(range(len(df))):
        input = df.question[i]
        correct_output = df.best_answer[i]
        
        input_pert = perturb.llm_prompting()
        # outs, inp_pps = produce_output_variations(inp, type_=args.perturb_type)
        output_pert = generate.a2c()
        # options, cons_inp_pps = produce_output_variations(inp, type_=args.perturb_type)
        # cons_outs = ans_via_comparison(inp, options, type_=args.perturb_type)
        
        all_input.extend([input]*len(input_pert))
        all_output.extend(output_pert)
        all_input_pert.extend([input_pert if args.perturb_type=="context" else ['']*len(input_pert)][0])
        # all_consistent_output.extend(cons_outs)
        # all_cons_inp_pps.extend([cons_inp_pps if args.perturb_type=="context" else ['']*len(outs)][0])
        all_correct_output.extend([correct_output]*len(input_pert))

        res_df = pd.DataFrame({
            "input": all_input,
            "input_perturb": all_input_pert,
            "output": all_output,
            "correct_outputs": all_correct_output,
        })
        
        res_df.to_csv(f"result_{args.model_name.replace('/', '')}_{args.perturb_type}", index=False)
    res_df.to_csv(f"result_{args.model_name.replace('/', '')}_{args.perturb_type}", index=False)