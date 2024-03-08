from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)

your_model_path = "Your path model"
chat_topic = "Machine Learning"
user_question = str(input("Enter your question : ")) 
template = """
Your name is Dico, you are machine learning expert and personal assistant with polite and wise. 
explain this question with complete and clear: '{question}' the topic is about {topic} and remember your name is Dico.
"""

prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(
    topic=chat_topic,
    question=user_question
)
CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=your_model_path,
    n_ctx=6000,
    n_gpu_layers=512,
    n_batch=30,
    callback_manager=CallbackManager,
    temperature=0.9,
    max_tokens=4095,
    n_parts=1,
    verbose=0
)
llm(final_prompt)
