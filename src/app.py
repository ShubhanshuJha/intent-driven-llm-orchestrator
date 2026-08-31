from llm_models.chat_models import ChatModel
from llm_models.agentic_models import AgentModel
from llm_models.light_models import UserIntentClassifierModel
from llm_models.coding_models import CodingModel
from llm_models.judge_models import LLMJudgeModel

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning,
)

import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


user_intent_classifier_model: UserIntentClassifierModel = None
available_models: list[str] = ['agent_model', 'chat_model', 'coding_model']
initialized_models: dict = {}
chat_model: ChatModel = None
agent_model: AgentModel = None
coding_model: CodingModel = None


def ask(query: str, model_name: str):
    global agent_model, chat_model, coding_model, initialized_models

    if model_name == 'agent_model' and not agent_model:
        import uuid
        agent_model = AgentModel(thread_id=f"session-{uuid.uuid4().hex[:5]}")
        initialized_models[available_models[0]] = agent_model
    if model_name == 'chat_model' and not chat_model:
        chat_model = ChatModel(memory_window=5)
        initialized_models[available_models[1]] = chat_model
    if model_name == 'coding_model' and not coding_model:
        coding_model = CodingModel(memory_window=20)
        initialized_models[available_models[2]] = coding_model
    return initialized_models[model_name].run(query=query)

def evaluate(query: str, response: str):
    judge_model = LLMJudgeModel()
    import time
    evaluation = judge_model.run(
        query=query,
        response=response,
    )
    print(f"(*) {type(evaluation) = }")
    print(f"\n(*) Score: {evaluation.score}/10")
    print(f"(*) Correctness: {evaluation.correctness}/10")
    print(f"(*) Relevance: {evaluation.relevance}/10")
    print(f"(*) Completeness: {evaluation.completeness}/10")
    print(f"(*) Instruction Following: {evaluation.instruction_following}/10")
    print(f"(*) Verdict: {evaluation.verdict}")
    print(f"(*) Critique: {evaluation.critique}")
    time.sleep(1)
    return evaluation.verdict.upper() == 'PASS'


def run(user_query: str, validate_and_fix: bool = True):
    global user_intent_classifier_model
    model_name = user_intent_classifier_model.run(query=user_query).strip()
    print(f"(*) Using {model_name} model for current query.")
    response = ask(query=user_query, model_name=model_name)
    if validate_and_fix:
        success = evaluate(query=user_query, response=response)
        if not success and validate_and_fix:
            run(user_query=user_query, validate_and_fix=False)
    print(f"(*) LLM Response: {response}")


if __name__ == '__main__':
    if not user_intent_classifier_model:
        user_intent_classifier_model = UserIntentClassifierModel(available_models=available_models, memory_window=5)
    while True:
        user_query = input("\nEnter your query [Write 'exit' to exit program]: ")
        if user_query.strip().lower() == 'exit':
            break
        run(user_query=user_query)
    
    # import time
    # test_queries = [
    #     "Who is the richest person in the world right now?",
    #     "Write a Python implementation of an LRU cache.",
    #     "Write a poem on nature in 4 lines",
    #     "Write Java code to demonstrate use of TreeSet."
    # ]
    # for query in test_queries:
    #     print(f"\n(*) Current Query: {query}")
    #     run(user_query=query)
    #     time.sleep(1)

