import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from llm_models.chat_models import ChatModel
from llm_models.agentic_models import AgentModel
from llm_models.light_models import UserIntentClassifierModel
from llm_models.coding_models import CodingModel

from llm_models.judge_models import LLMJudgeModel
from llm_models.llm_factory import LLMFactory

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning,
)


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

def run(user_query: str):
    global user_intent_classifier_model
    model_name = user_intent_classifier_model.run(query=user_query).strip()
    print(f"(*) Using {model_name} model for current query.")
    response = ask(query=user_query, model_name=model_name)
    print(f"(*) LLM Response: {response}")
    return response

def evaluate(query: str, response: str):
    global judge_model
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


if __name__ == '__main__':
    continuous_test = False
    judge_model: LLMJudgeModel = LLMJudgeModel()

    if not user_intent_classifier_model:
        user_intent_classifier_model = UserIntentClassifierModel(available_models=available_models, memory_window=5)
    if continuous_test:
        while True:
            user_query = input("\nEnter your query [Write 'exit' to exit program]: ")
            if user_query.strip().lower() == 'exit':
                break
            run(user_query=user_query)
    else:
        import time
        test_queries = [
            "Who is the richest person in the world right now?",
            "Write a Python code to show implementation of an LRU cache.",
            "Write a poem on nature in 4 lines",
            "Write Java code to demonstrate use of TreeSet.",
            "Who is the richest person in the world right now?",
            "Write a formal DE application mail for me."
        ]
        for query in test_queries:
            print(f"\n(*) Current Query: {query}")
            response = run(user_query=query)
            evaluate(query=query, response=response)


