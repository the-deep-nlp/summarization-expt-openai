import os
import evaluate
from rouge_score import rouge_scorer
from enum import Enum
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import LLMSummarizationCheckerChain

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

rouge = evaluate.load('rouge')

class ChainTypes(Enum):
    """ Chain Types """
    STUFF = 0
    MAP_REDUCE = 1
    REFINE = 2

class Summarization:
    """ Summarization Module """
    def __init__(
        self,
        temperature=0.1,
        max_tokens=512,
        model_name="text-davinci-003" # gpt-3.5-turbo
    ):
        self.llm = OpenAI(
            temperature=temperature,
            model=model_name,
             max_tokens=max_tokens,
        )
    
    def textsplitter(self):
        """ Text Splitter """
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "\t"]
        )
    
    def load_data(self, filename):
        """ Load file from disk """
        with open(filename) as f:
            texts = f.read()
        return texts
    
    def create_docs(self, texts):
        """ Create Docs """
        text_splitter = self.textsplitter()
        texts = text_splitter.split_text(texts)
        docs = [Document(page_content=t) for t in texts]
        return docs
    
    def generate_prompt(self):
        """ Generate Prompts """
        prompt_template = """You are a humanitarian analyst and has a strong domain knowledge. 
        Write a concise summary of the following including key points:

        {text}

        CONCISE SUMMARY:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        return prompt

    def generate_refine_prompt(self):
        """ Generate Refine Prompt """
        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary\n"
            "(only if needed) with some more context below.\n"
            "----------------\n"
            "{text}\n"
            "----------------\n"
            "Given the new context, refine the original summary\n"
            "If the context isn't useful, return the original summary"
        )
        refine_prompt = PromptTemplate(
            template=refine_template,
            input_variables=["existing_answer", "text"]
        )
        return refine_prompt

    def generate_summary(
        self,
        docs,
        prompt,
        chain_type=ChainTypes.STUFF,
        verbose=False
    ):
        """ Generate Summary """
        if chain_type==ChainTypes.MAP_REDUCE:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                verbose=verbose,
                map_prompt=prompt,
                combine_prompt=prompt
            )
        elif chain_type==ChainTypes.REFINE:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="refine",
                verbose=verbose,
                question_prompt=prompt,
                refine_prompt=self.generate_refine_prompt()
            )
        else:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type=chain_type,
                verbose=verbose
            )
        return chain.run(docs) # summary

    def use_summ_checker_chain(self, texts):
        """ Use in-built summarization checker chain """
        checker_chain = LLMSummarizationCheckerChain(
            llm=self.llm,
            verbose=False,
            max_checks=2
        )
        return checker_chain.run(texts)

    def evaluate(
        self,
        original_summary,
        generated_summary
    ):
        """ Route Evaluation """
        metric_score = rouge.compute(
            predictions=generated_summary,
            references=original_summary,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )
        return metric_score

    def rouge_scorer(
        self,
        original_summary,
        model_summary
    ):
        """ Calculate rouge score """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(original_summary, model_summary)
        return scores