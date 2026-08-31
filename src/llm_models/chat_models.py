from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

from llm_models.llm_factory import LLMFactory


CHAT_SYSTEM_PROMPT = """
You are a helpful, knowledgeable, and conversational AI assistant.

Your responsibilities:
- Answer the user's questions accurately and clearly in less than 2048 tokens.
- Maintain relevant context from the conversation history.
- Follow information explicitly provided by the user.
- Do not fabricate facts when information is unknown.
- Provide concise answers by default unless the user asks for detail.
- For technical questions, provide precise and technically correct explanations.
- For writing tasks, follow the requested style, tone, format, and length.

Conversation History:
{history}

User:
{input}

Assistant:
"""


class ChatModel:
    def __init__(self, memory_window: int = 10):
        self.memory_window = memory_window
        self.__model = self.__initialize_llm_model()
        self.__llm_memory = self.__get_memory(k=memory_window)
        self.__llm_chain = self.__build_chain()

    def __initialize_llm_model(self):
        return LLMFactory.get_model(
            profile="general_chat"
        )

    def __get_memory(self, k: int):
        return ConversationBufferWindowMemory(
            k=k,
            memory_key="history",
            input_key="input",
        )

    def __build_chain(self):
        prompt = PromptTemplate(
            input_variables=[
                "history",
                "input",
            ],
            template=CHAT_SYSTEM_PROMPT,
        )

        return ConversationChain(
            llm=self.__model,
            memory=self.__llm_memory,
            prompt=prompt,
            verbose=False,
        )

    def run(self, query: str) -> str:
        response = self.__llm_chain.invoke(
            {
                "input": query
            }
        )

        return response["response"]
