import openai
import pandas as pd  # Importa pandas y utiliza 'pd' como abreviatura
import os
import matplotlib
from dotenv import load_dotenv
import tiktoken
import langchain

from fastapi import FastAPI


#from models.model import Prompt

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline



from transformers import AutoModelForCausalLM, AutoTokenizer
"""
# Load the model and tokenizer
model_name = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text
input_text = "Cual es el numero de la factura electronica"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text using the model
output = model.generate(input_ids, max_length=100)

# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
"""

llm = HuggingFacePipeline.from_model_id(model_id="OpenAssistant/stablelm-7b-sft-v7-epoch-3", task="text-generation", model_kwargs={"temperature": 0.0, "max_length": 2048, 'device_map': 'auto',"offload_folder":'store'})

loader=PyPDFLoader("./E54051222045753R001297402700.pdf")

pages=loader.load_and_split()
#un elemento por cada página
#print(pages[2].page_content)

#Objeto que hace cortes en el texto
split=CharacterTextSplitter(chunk_size=500,separator='.\n')
textos=split.split_documents(pages)
#print(textos)
#print(len(textos))


embeddings = HuggingFaceEmbeddings()


query_result=embeddings.embed_query(textos[0].page_content)
#print(query_result)

vectorstore=Chroma.from_documents(textos,embeddings)

qa=ConversationalRetrievalChain.from_llm(llm,vectorstore.as_retriever(),return_source_documents=True)

chat_history=[]
query="Cual es el numero de la factura electronica"
result=qa({"question":query,"chat_history":chat_history})
result["answer"]

#Extraemos la parte de page_content de cada texto y lo pasamos a un dataframe

textos=[str(i.page_content) for i in textos]
parrafos= pd.DataFrame(textos,columns=["texto"])
#print(parrafos)

tokenizer=tiktoken.get_encoding("cl100k_base")
parrafos['n_tokens']=parrafos.texto.apply(lambda x: len(tokenizer.encode(x)))
print(parrafos.sample())

#embeddings


#parrafos['Embedding']=parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
#print(parrafos)



app=FastAPI()
load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')