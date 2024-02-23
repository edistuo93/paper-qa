import os
import argparse
import pickle

from paperqa.contrib import ZoteroDB
from paperqa import Docs, LangchainVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="Tool for processing queries with an optional chat mode.")
parser.add_argument("--chat", '-c', action="store_true", help="Enable chat mode")
parser.add_argument("--prompt", '-p', type=str, help="Query to process")
parser.add_argument("--model", '-m', type=str, default='gpt-3.5-turbo', help="Query to process")
args = parser.parse_args()
if not args.chat: assert args.prompt, 'Must specify --chat for chat mode or give a prompt with --prompt.'

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
else:
    raise ValueError(f'Must build semantic search database with model {args.model} '
                      'with build_search_index.py before querying!')


# chat with LLM
if args.chat:
    prompt = args.prompt if args.prompt else input('Prompt ("q" to quit): ')
    print('\n', docs.query(prompt), '\n')
    while True:
        prompt = input('Prompt: ')
        if (prompt.lower() in ['q', 'quit']):
            break
        print('\n', docs.query(prompt), '\n')
else:
    print(docs.query(args.prompt))

with open(PAPERQA_CHECKPOINT, "wb") as f:
    pickle.dump(docs, f)
