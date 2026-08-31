from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from llm_models.llm_factory import LLMFactory


USER_INTENT_CLASSIFIER_SYSTEM_PROMPT: str = """
Requirement:
Classify the user's request into exactly one model from:
{available_models}

Task:
Choose the model based on the capability required to fulfill the request.

- agent_model:
  Use when external tools or external information are required, including
  current/live information, web retrieval, exact calculations, current
  date/time, or interaction with external systems.

- chat_model:
  Use for conversation, explanations, reasoning, summarization, brainstorming,
  creative writing, professional writing, essays, poems, stories, rewriting,
  and other natural-language tasks that do not require external tools.

- coding_model:
  Use only when the primary request involves software engineering or
  programming artifacts, such as source code, scripts, queries, configuration,
  debugging, refactoring, code review, tests, APIs, or implementation design.

Input:
A single user query.

Output:
Return exactly one model name from {available_models}.

Note:
Classify by the requested outcome, not by individual words.

The words "write", "create", "generate", or "design" do not imply coding_model
unless the requested output is code or a software-related artifact.

Natural-language or creative content belongs to chat_model.
Current or externally changing information belongs to agent_model.
Programming and software implementation belongs to coding_model.

Do not answer the user's request.
Do not explain your decision.
Return only the selected model name.
"""


class UserIntentClassifierModel:
    def __init__(self, available_models: list[str], memory_window: int = 3):
        self.memory_window = memory_window
        self.available_models = available_models
        self.__model = self.__initialize_llm_model()
        self.__llm_memory = self.__get_memory(k=memory_window)
        self.__llm_chain = self.__build_chain()

    def __initialize_llm_model(self):
        return LLMFactory.get_model(
            profile="fast_lightweight"
        )

    def __get_memory(self, k: int):
        return ConversationBufferWindowMemory(
            k=k,
            memory_key="history",
            input_key="input",
        )

    def __build_chain(self):
        # prompt = PromptTemplate(
        #     input_variables=[
        #         "history",
        #         "input",
        #     ],
        #     partial_variables={
        #         "available_models": ", ".join(self.available_models)
        #     },
        #     template=USER_INTENT_CLASSIFIER_SYSTEM_PROMPT,
        # )
        # return ConversationChain(
        #             llm=self.__model,
        #             memory=self.__llm_memory,
        #             prompt=prompt,
        #             verbose=False,
        #         )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    USER_INTENT_CLASSIFIER_SYSTEM_PROMPT
                ),
                (
                    "human",
                    "{input}"
                ),
            ]
        )
        prompt = prompt.partial(
            available_models=", ".join(
                self.available_models
            )
        )
        return prompt | self.__model

    def run(self, query: str) -> str:
        response = self.__llm_chain.invoke(
            {
                "input": query
            }
        )
        model_name = response.content.strip()
        if model_name not in self.available_models:
            raise ValueError(
                f"Invalid model classification: {model_name}"
            )
        return model_name
