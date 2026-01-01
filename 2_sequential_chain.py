from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT']='sequential LLm app'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatCerebras(model='qwen-3-32b')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
config = {
    'run_name': 'sequential chain',
    'tags': ['llm_app', 'report generation', 'summarization'],
    'metadata': {
        'model': 'quen-3-32b',
        'model_temp': 0.7,
        'parser': 'stroutputparser'
    }
}

result = chain.invoke(
    {'topic': 'Unemployment in India'},
    config=config
)

print(result)
