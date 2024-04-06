from typing import List
import math
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
# Uncomment to see all the LLM interactions happening under the hood
# import langchain ; langchain.debug = True

LLM = ChatOpenAI(max_tokens=1500)

def create_arguement_participant(
    name: str,
    traits: str,
    status: str,
    experiences: List[str],
    llm: BaseLanguageModel = LLM,
) -> GenerativeAgent:
    """Creates a generative agent with the given traits and experiences."""

    print(f"Creating Agent: {name}...")
    memory_retriever = create_time_weighted_memory_retriever()
    memory = GenerativeAgentMemory(
        memory_retriever=memory_retriever,
        reflection_threshold=5,
        verbose=False,
        llm=llm,
    )
    agent = GenerativeAgent(
        name=name,
        traits=traits,
        status=status,
        llm=llm,
        memory=memory,
        verbose=False,
    )
    for experience in experiences:
        agent.memory.add_memory(f"{name} {experience}")
    # Secret ingredient that keeps the argument going
    agent.memory.add_memory(
        f"{name} is very opinionated and likes to disagree with people whenever possible"
    )
    print(agent.get_summary() + "\n" + "-" * 20)
    return agent


def start_argument(
    agents: List[GenerativeAgent], conversation_starter: str, max_turns: int = 10
) -> None:
    """Initiates an argument giving each agent a chance to react respond."""
    print("-" * 20 + f"\033[96m{conversation_starter}\033[00m")

    # Start the argument started off by getting a reaction from each agent
    for agent in agents:
        _, observation = agent.generate_reaction(conversation_starter)
        print(observation)

    # Get the argument going
    turns, keep_talking = 0, True
    while True:
        for agent in agents:
            keep_talking, observation = agent.generate_dialogue_response(observation)
            print(observation)
            if not keep_talking:
                return
        turns += 1
        if turns >= max_turns:
            return


def create_time_weighted_memory_retriever() -> TimeWeightedVectorStoreRetriever:
    """Creates a retriever that priorizes recent memories."""
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


vincent = create_arguement_participant(
    name="Vincent Vega",
    traits="Cool, collected, career criminal, an addict and hitman.",
    status="Hungry",
    experiences=[
        "recently came back from europe",
        "like burgers and milk shakes",
        "goes to the bathroom a lot",
        "always has something happen when he comes out of the bathroom",
        "is a likable and sensitive person.",
        "talks like a stoner",
    ],
)
jules = create_arguement_participant(
    name="Jules Winnfield",
    traits="Dangerous, spiritual, and short-tempered killer.",
    status="Hungry",
    experiences=[
        "has a commanding presence, exudes confidence and authority in his actions.",
        "recently had a near death experience that has made him question everything",
        "likes to quote ancient scriptures",
        "argues a lot with his friend Vincent Vega",
        "talks like a character from a blaxploitation movie",
        "does not like pork",
    ],
)
winston = create_arguement_participant(
    name="Winston Wolf",
    traits="Direct, decisive, and demanding crime fixer",
    status="Hungry",
    experiences=[
        "is highly respected by other criminals",
        "just left a funeral",
        "specializes in solving problems and clean up messy situations",
        "is professional, confident, and has calm demeanor under pressure.",
        "is polite and courteous in his interactions.",
    ],
)

agents = [vincent, jules, winston]
start_argument(agents, f"Who here likes bacon!?", max_turns=10)
