import spacy
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .prompt_template import EVAL_STEP1_TEMPLATE, EVAL_STEP2_TEMPLATE


class PP_Detector:
    """
    This class implements a Paraphrase Detector using a finetuned LLM. Currently, it uses [DeBERTa V3 finetuned on the PAWS dataset](https://huggingface.co/domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector).
    It serves to assess if two given text segments are paraphrases of each other.
    The model outputs a score between 0 and 1, indicating the likelihood of the inputs being paraphrases.

    Attributes:
        detection_tokenizer (AutoTokenizer): Hugging Face tokenizer, initialized from a pretrained model specified by `tok_path`.
        detection_model (AutoModelForSequenceClassification): Hugging Face model fine-tuned for paraphrase detection, loaded from `model_path`.

    Methods:
        __init__(tok_path, model_path, max_len): Initializes the Paraphrase Detector.
        score_binary(sentence1, sentence2): Calculates the paraphrase probability scores for two input sentences.

    Parameters:
        tok_path (str): Path or identifier for the tokenizer to be used.
                        Default is set to "domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector".
        model_path (str): Path or identifier for the model to be used.
                          Mirrors the default of `tok_path`.
        max_len (int): Maximum length of the tokenized input. Default value is 30.

    Example:
        ```
        detector = PP_Detector()
        score_no_para, score_para = detector.score_binary("This is a sentence.", "This is another sentence.")
        ```
        `score_no_para` represents the probability of the inputs not being paraphrases, while `score_para` represents the probability of them being paraphrases.
    """

    def __init__(
        self,
        tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",
        model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",
        max_len=30,
    ):
        super(PP_Detector, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.detection_model.to("cuda")

    def score_binary(self, sentence1, sentence2):
        """
        Calculates the paraphrase probability scores for two input sentences.

        Parameters:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            (float): Probabilities that the two sentences are not each other's paraphrases.
            (float): Probabilities that the two sentences are each other's paraphrases.
        """
        inputs = self.detection_tokenizer(
            sentence1, sentence2, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probabilites and scores for not paraphrase and paraphrase
        return scores.T[0].item(), scores.T[1].item()


class NLI:
    """
    Allows for determining it two sentences contradict
    each other or if one sentence entails the other.

    Parameters:
        tok_path (str): Path to the tokenizer.
        model_path (str): Path to the model.
        max_length (int): Maximum length of the input sequences.
    """

    def __init__(
        self,
        tok_path="microsoft/deberta-base-mnli",
        model_path="microsoft/deberta-base-mnli",
        max_len=30,
    ):
        super(NLI, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.detection_model.to("cuda")

    def entailed(self, sentence1, sentence2):
        """
        Determines if the first sentence entails the second.

        Parameters:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            (float): Probability that sentence1 entails sentence2.
        """
        inputs = self.detection_tokenizer(
            sentence1, sentence2, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[2].item()

    def contradicted(self, sentence1, sentence2):
        """
        Determines if the first sentence contradicts the second.

        Parameters:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            (float): Probability that sentence1 contradicts sentence2.
        """
        inputs = self.detection_tokenizer(
            sentence1, sentence2, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[0].item()


class Agreement:
    """
    The Agreement class is designed to assess the similarity or agreement between two text outputs.
    It supports a number of evaluation metrics: BLEU, BERTScore, paraphrase detection, entailment,
    contradiction, named entity recognition (NER), and LLM based comparison.

    Parameters:
        agreement_name (str): The name of the agreement metric to use.
        aux_model: An auxiliary LLM required for certain agreement metrics.
    """

    def __init__(self, agreement_name, aux_model):
        super(Agreement, self).__init__()
        if agreement_name.lower() == "bleu":
            self.bleu = evaluate.load("bleu")
            self.agreement_fn = self.bleu_agreement
        elif agreement_name.lower() == "bertscore":
            self.bertscore = evaluate.load("bertscore")
            self.agreement_fn = self.bertscore_agreement
        elif agreement_name.lower() == "paraphrase_detector":
            self.pp_detector = PP_Detector()
            self.agreement_fn = self.pp_agreement
        elif agreement_name.lower() == "entailment":
            self.nli = NLI()
            self.agreement_fn = self.entailment_agreement
        elif agreement_name.lower() == "contradiction":
            self.nli = NLI()
            self.agreement_fn = self.contradiction_agreement
        elif agreement_name.lower() == "ner":
            self.NER = spacy.load("en_core_web_sm")
            self.agreement_fn = self.ner_agreement
        elif agreement_name.lower() == "llm":
            # pipe = pipeline(model="google/flan-t5-xl", device_map="auto")
            # llm = HuggingFacePipeline(pipeline=pipe)
            # step 1
            prompt_eval_step1 = PromptTemplate(
                input_variables=["context", "question"],
                template=EVAL_STEP1_TEMPLATE,
            )
            self.chain_step1 = LLMChain(llm=aux_model, prompt=prompt_eval_step1)
            self.chain_step1.verbose = False
            # step 2
            prompt_eval_step2 = PromptTemplate(
                input_variables=["question", "answer1", "answer2"],
                template=EVAL_STEP2_TEMPLATE,
            )
            self.chain_step2 = LLMChain(llm=aux_model, prompt=prompt_eval_step2)
            self.chain_step2.verbose = False
            self.agreement_fn = self.llm_agreement
        else:
            raise Exception(f"agreement name '{agreement_name}' not available")

    def bleu_agreement(self, output_i, output_j):
        """
        Calculates the BLEU score to evaluate the agreement between two text outputs.

        Parameters:
            output_i (str): The first output text to compare.
            output_j (str): The second output text to compare.

        Returns:
            (float): The BLEU score indicating the similarity between output_i and output_j.
        """
        if not output_i:
            return 0
        if not output_j:
            return 0
        bleu_score = self.bleu.compute(predictions=[output_i], references=[output_j])
        return bleu_score["bleu"] or 0.0

    def bertscore_agreement(self, output_i, output_j):
        """
        Computes the BERTScore, a metric for evaluating the agreement between two textual outputs.

        Parameters:
            output_i (str): The first output text for comparison.
            output_j (str): The second output text for comparison.

        Returns:
            (float): The BERTScore representing the agreement between output_i and output_j.
        """
        bertscore_score = self.bertscore.compute(
            predictions=[output_i], references=[output_j], lang="en"
        )
        return bertscore_score["f1"][0]

    def pp_agreement(self, output_i, output_j):
        """
        Uses a paraphrase detector to evaluate the agreement between two text outputs.

        Parameters:
            output_i (str): The first output text for comparison.
            output_j (str): The second output text for comparison.

        Returns:
            (float): A score indicating whether the outputs are paraphrases of each other.
        """
        pp_detector_score = self.pp_detector.score_binary(output_i, output_j)
        return pp_detector_score[1]

    def entailment_agreement(self, output_i, output_j):
        """
        Assesses if one text output entails the other using a natural language inference model.

        Parameters:
            output_i (str): The first output text for comparison.
            output_j (str): The second output text for comparison.

        Returns:
            (float): A score indicating whether one output entails the other.
        """
        return self.nli.entailed(output_i, output_j)

    def contradiction_agreement(self, output_i, output_j):
        """
        Evaluates whether two text outputs contradict each other using a natural language inference model.

        Parameters:
            output_i (str): The first output text for comparison.
            output_j (str): The second output text for comparison.

        Returns:
            (float): A score indicating whether one output contradicts the other.
        """
        return self.nli.contradicted(output_i, output_j)

    def ner_agreement(self, output_i, output_j):
        """
        Uses Named Entity Recognition (NER) to evaluate the agreement between two text outputs based on the matching entities.

        Parameters:
            output_i (str): The first output text for entity comparison.
            output_j (str): The second output text for entity comparison.

        Returns:
            (float): A score based on the proportion of matching entities in the two outputs.
        """
        pro_texti = self.NER(output_i)
        pro_textj = self.NER(output_j)
        num_matches = 0
        all_NERs = []
        for word_i in pro_texti.ents:
            for word_j in pro_textj.ents:
                all_NERs.extend([word_i.text, word_j.text])
                if word_i.text == word_j.text:
                    num_matches += 1
                    break  # no multiple match
        if len(all_NERs) == 0:
            return 0.0
        return float(num_matches / len(set(all_NERs)))

    def llm_agreement(self, input, output_i, output_j):
        """
        Utilizes a large language model (LLM) to assess the agreement between two outputs based on a specific input.
        This method involves two steps: first, it uses the LLM to process each output with the input;
        second, it compares the LLM-processed outputs to determine if they are in agreement.

        Parameters:
            input (str): The original input text based on which the outputs were generated.
            output_i (str): The first output text for comparison.
            output_j (str): The second output text for comparison.

        Returns:
            (int): Returns 1 if the LLM determines the outputs are in agreement based on the input, otherwise 0.
        """
        out1_step1 = self.chain_step1.run({"context": output_i, "question": input})
        out2_step1 = self.chain_step1.run({"context": output_j, "question": input})

        score = self.chain_step2.run(
            {
                "question": input.strip(),
                "answer1": out1_step1.strip(),
                "answer2": out2_step1.strip(),
            }
        )
        return 1 if score.strip() == "Yes" else 0
