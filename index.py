from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Normal CR
CR_Template_Normal = TextLoader('CR_Template_Normal.txt').load()
text_splitter_CR_Template_Normal = CharacterTextSplitter(chunk_overlap=100)
CR_Template_Normal_content = text_splitter_CR_Template_Normal.split_documents(CR_Template_Normal)

# Standard CR
CR_Template_Standard = TextLoader('CR_Template_Standard.txt').load()
text_splitter_CR_Template_Standard = CharacterTextSplitter(chunk_overlap=100)
CR_Template_Standard_content = text_splitter_CR_Template_Standard.split_documents(CR_Template_Standard)

Actual_CR = TextLoader('CR00001234.txt').load()
text_splitter_Actual_CR = CharacterTextSplitter(chunk_overlap=100)
Actual_CR_content = text_splitter_Actual_CR.split_documents(Actual_CR)

faiss_db1 = FAISS.from_documents(CR_Template_Normal_content, embedding_model)
faiss_db2 = FAISS.from_documents(CR_Template_Standard_content, embedding_model)
faiss_db3 = FAISS.from_documents(Actual_CR_content, embedding_model)

faiss_db1.merge_from(faiss_db2)
faiss_db1.merge_from(faiss_db3)

retriever = faiss_db1.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = AzureChatOpenAI(
    temperature=0,
    deployment_name="gpt-4",
)

prompt_template = """

You have been provided with two distinct CR templates: STANDARD and NORMAL. 
The STANDARD CR template is marked with the identifiers "THIS IS STANDARD CR TEMPLATE DOCUMENT - START" at the beginning and "THIS IS STANDARD CR TEMPLATE DOCUMENT - END" at the end. 
Similarly, the NORMAL CR template is marked with "THIS IS NORMAL CR TEMPLATE DOCUMENT - START" and "THIS IS NORMAL CR TEMPLATE DOCUMENT - END" as its respective start and end points.

Additionally, you have an actual CR Document, which commences with "THIS IS AN ACTUAL CR DOCUMENT - START" and concludes with "THIS IS AN ACTUAL CR DOCUMENT - END."

Your task is to determine whether the actual CR Document conforms to the STANDARD or NORMAL template, without mixing elements from both. 
Please examine carefully and provide responses to the following user inquiries regarding this task:

Context:
{context}

User question: 
{question}

Respond to the user using JSON format:
"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=['context', 'question']
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type_kwargs={"prompt": QA_PROMPT},
    verbose=True
)

question = """
What is the CR Category? 
"""
result = qa_chain({"query": question})
print(result)

question = """

Extract the following information from the given context: 

- Background/justification of this CR
- Man hours saved information
- The date & time when the CR will be deployed
- Sanity date & time
- Rollback/backout window date & time
- Persons in charge
- List of the recipes & folders 
- Communication plan, and whether the communications are required or not. 
- SIT and UAT document
- Impacted services
- Post deployment with person name in charge in a summary

"""

result = qa_chain({"query": question})
print(result)
