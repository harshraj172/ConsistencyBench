import numpy as np
import spacy
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompt_templates import *

class PP_Detector():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", max_len=30):
        super(PP_Detector, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def score_binary(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probabilites and scores for not paraphrase and paraphrase
        return scores.T[0].item(), scores.T[1].item()


class NLI():
    """
    microsoft/deberta-v2-xxlarge-mnli uses
    "id2label": {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT"
      },
    """
    def __init__(self, tok_path="microsoft/deberta-base-mnli", model_path="microsoft/deberta-base-mnli", max_len=30):
        super(NLI, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def entailed(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[2].item()

    def contradicted(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[0].item()
    

class SemanticAgreement():
    def __init__(self, agreement_name):
        super(SemanticAgreement, self).__init__()
        if agreement_name.lower()=="bleu":
            self.bleu = evaluate.load("bleu")
            self.agreement_fn = self.bleu_agreement
        elif agreement_name.lower()=="bertscore":
            self.bertscore = evaluate.load("bertscore")
            self.agreement_fn = self.bertscore_agreement
        elif agreement_name.lower()=="paraphrase_detector":
            self.pp_detector = PP_Detector(tok_path, model_path)
            self.agreement_fn = self.pp_agreement
        elif agreement_name.lower()=="entailment":
            self.nli = NLI()
            self.agreement_fn = self.entailment_agreement
        elif agreement_name.lower()=="contradiction":
            self.nli = NLI()
            self.agreement_fn = self.contradiction_agreement
        elif agreement_name.lower()=="ner":
            self.NER = spacy.load("en_core_web_sm")
            self.agreement_fn = self.ner_agreement
        elif agreement_name.lower()=="llm":
            pipe = pipeline(model="google/flan-t5-xl", device_map="auto")
            llm = HuggingFacePipeline(pipeline=pipe)
            # step 1
            prompt_eval_step1 = PromptTemplate(
                    input_variables=["context", "question"],
                    template=EVAL_STEP1_TEMPLATE,)
            self.chain_step1 = LLMChain(llm=llm, prompt=prompt_eval_step1)    
            self.chain_step1.verbose = False    
            # step 2
            prompt_eval_step2 = PromptTemplate(
                    input_variables=["question", "answer1", "answer2"],
                    template=EVAL_STEP2_TEMPLATE,)
            self.chain_step2 = LLMChain(llm=llm, prompt=prompt_eval_step2)    
            self.chain_step2.verbose = False

    def bleu_agreement(self, input, output_i, output_j):
        if not output_i:
            return 0
        if not output_j:
            return 0
        bleu_score = self.bleu.compute(predictions=[output_i], references=[output_j])
        return bleu_score['bleu'] or 0.0
    
    def bertscore_agreement(self, input, output_i, output_j):
        bertscore_score = self.bertscore.compute(predictions=[output_i], references=[output_j], lang='en')
        return bertscore_score['f1'][0]

    def pp_agreement(self, input, output_i, output_j):
        pp_detector_score = self.pp_detector.score_binary(output_i, output_j)
        return pp_detector_score[1]

    def entailment_agreement(self, input, output_i, output_j):
        return self.nli.entailed(output_i, output_j)

    def contradiction_agreement(self, input, output_i, output_j):
        return self.nli.contradicted(output_i, output_j)

    def ner_agreement(self, input, output_i, output_j):
        pro_texti = self.NER(output_i)
        pro_textj = self.NER(output_j)
        num_matches = 0
        all_NERs = []
        for word_i in pro_texti.ents:
            for word_j in pro_textj.ents:
                all_NERs.extend([word_i.text, word_j.text])
                if word_i.text == word_j.text:
                    num_matches += 1
                    break # no multiple match
        if len(all_NERs) == 0:
            return 0.0
        return float(num_matches/len(set(all_NERs)))
    
    def llm_agreement(self, input, output_i, output_j):
        out1_step1 = self.chain_step1.run({"context":output_i, "question":input})
        out2_step1 = self.chain_step1.run({"context":output_j, "question":input})

        score = self.chain_step2.run({"question":input.strip(), "answer1":out1_step1.strip(), "answer2":out2_step1.strip()})
        return 1 if score.strip()=='Yes' else 0