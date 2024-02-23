import os
import argparse
import pickle

from paperqa.contrib import ZoteroDB
from paperqa import Docs, LangchainVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="Tool for processing queries with an optional chat mode.")
parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="Query to process")
args = parser.parse_args()

# load checkpoint
PAPERQA_DIR = os.path.expanduser('~') + '/.paperqa_docs'
PAPERQA_CHECKPOINT = PAPERQA_DIR + f"/paperqa_docs_{args.model}.pkl"
if not os.path.exists(PAPERQA_DIR):
    os.mkdir(PAPERQA_DIR)
if os.path.exists(PAPERQA_CHECKPOINT):
    with open(PAPERQA_CHECKPOINT, "rb") as f:
        docs = pickle.load(f)
        docs.set_client()
        print('Loaded paper-qa checkpoint from', PAPERQA_CHECKPOINT)


# broken attempt at using LlamaCpp
# from langchain.llms import LlamaCpp
# from langchain import PromptTemplate, LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain.embeddings import LlamaCppEmbeddings
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# model_path = 'llama-2-7b.Q2_K.gguf'  #'llama-7b.ggmlv3.q2_K.bin'
# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path=model_path, callbacks=[StreamingStdOutCallbackHandler()]
# )
# embeddings = LlamaCppEmbeddings(model_path=model_path)
# # docs = Docs(llm=llm, embeddings=embeddings)  # having issues with LlamaCpp models in paper-qa

# separate issue: I'm not correctly incorporating the external vector DB; 
# however, Docs caches embeddings already so not needed
# my_index = LangchainVectorStore(cls=FAISS, embedding_model=OpenAIEmbeddings())

# update documents
docs = Docs(llm=args.model)
zotero = ZoteroDB(library_type="group")  # "user" if user library
for item in zotero.iterate(limit=5000):
    # if item.num_pages > 30:
    #     continue  # skip long papers
    docs.add(item.pdf, docname=item.key)

with open(PAPERQA_CHECKPOINT, "wb") as f:
    pickle.dump(docs, f)
