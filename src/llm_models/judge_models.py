from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_models.llm_factory import LLMFactory


class LLMJudgeResponse(BaseModel):
    score: float = Field(
        ge=0,
        le=10,
        description="Overall quality score."
    )

    correctness: float = Field(
        ge=0,
        le=10,
        description="Factual and technical correctness."
    )

    relevance: float = Field(
        ge=0,
        le=10,
        description="Relevance to the user's request."
    )

    completeness: float = Field(
        ge=0,
        le=10,
        description="How completely the response satisfies the request."
    )

    instruction_following: float = Field(
        ge=0,
        le=10,
        description="How well the response follows the user's instructions."
    )

    verdict: Literal["PASS", "FAIL"] = Field(
        description="Final evaluation verdict."
    )

    critique: str = Field(
        description="Brief and specific explanation of the evaluation."
    )


LLM_JUDGE_SYSTEM_PROMPT: str = """
Requirement:
Evaluate an LLM-generated response against the user's request.

Task:
Judge the response objectively based on:
- correctness
- relevance
- completeness
- instruction-following
- overall quality

Input:
You will receive the original user query, candidate response, and optional
supporting context.

Output:
Return ONLY the structured evaluation using the following format:

{format_instructions}

Note:
Do not answer the original user query.
Do not rewrite the candidate response.
Do not return markdown.
Do not return a table.
Do not return a code block.
Do not include text before or after the structured output.
Do not assume unsupported information is correct.
Use scores between 0 and 10.
Use PASS only when the response adequately satisfies the request.
Keep the critique brief and specific.
"""


LLM_JUDGE_INPUT_PROMPT: str = """
User Query:
{query}

Candidate Response:
{response}

Supporting Context:
{context}
"""


class LLMJudgeModel:

    def __init__(self):
        self.__model = self.__initialize_llm_model()
        self.__output_parser = self.__get_output_parser()
        self.__llm_chain = self.__build_chain()

    def __initialize_llm_model(self):
        return LLMFactory.get_model(
            profile="judge"
        )

    def __get_output_parser(self):
        return PydanticOutputParser(
            pydantic_object=LLMJudgeResponse
        )

    def __build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    LLM_JUDGE_SYSTEM_PROMPT
                ),
                (
                    "human",
                    LLM_JUDGE_INPUT_PROMPT
                ),
            ]
        )

        prompt = prompt.partial(
            format_instructions=(
                self.__output_parser.get_format_instructions()
            )
        )

        return (
            prompt
            | self.__model
            | self.__output_parser
        )

    def run(self, query: str, response: str, context: str = "") -> LLMJudgeResponse:

        return self.__llm_chain.invoke(
            {
                "query": query,
                "response": response,
                "context": context,
            }
        )
