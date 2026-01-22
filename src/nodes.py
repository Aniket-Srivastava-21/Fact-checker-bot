from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.cache import store_claim_cache, retrieve_similar_claims
from src.tools import tools
from src.state import LLMResponse, FactCheckState
from src.utils import embed_text, get_final_response_with_structured_output, get_retrieved_data, llm_decide_reuse, client


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(LLMResponse)

VERIFY_PROMPT = """You are a fact-checking assistant. Use the tools available to gather information and verify claims.
            Claim: {claim}
            Provide a concise verdict only based on the evidence you gather. If you cannot verify the claim, respond with 'Insufficient information to verify the claim.'"""


def embed_node(state: FactCheckState) -> FactCheckState:
    claim = state['claim']
    print(f"Input claim: {claim}")

    embedding = embed_text(claim)

    return {'messages': [HumanMessage(content=VERIFY_PROMPT.format(claim=claim))], 'embedding': embedding}

def cache_lookup(state: FactCheckState) -> FactCheckState:
    print("Checking cache...")

    retrieved = retrieve_similar_claims(state['embedding'])

    if retrieved and llm_decide_reuse(state['claim'], retrieved):
        print("Cache hit!")
        return {'cache_hit': True, 'cached_results': retrieved}
    else:
        print("Cache miss!")
        return {'cache_hit': False}

def reuse_verdict_node(state):
    retrieved = state["cached_results"]

    prompt = f"""Based on these previous data:

            {get_retrieved_data(retrieved)}
    
            Answer the claim below:
            {state["claim"]}
            """

    output = structured_llm.invoke(prompt)

    response = get_final_response_with_structured_output(output)

    return {'messages': [response], 'confidence': output.confidence, 'verdict': output.verdict, 'evidence': output.evidence}


def verify_claim(state: FactCheckState) -> FactCheckState:
    print("Verifying claim...")

    messages = state['messages']

    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}

def formatting_node(state: FactCheckState) -> FactCheckState:
    print("Formatting fact-check response...")
    
    # We pass the conversation history so it can extract the final answer
    prompt = f"""
                Based on the fact-check messages given below:
                {state['messages']}

                Respond in the following format:
                {LLMResponse.model_json_schema()}
            """

    structured_response: LLMResponse = structured_llm.invoke(prompt)

    final_response = get_final_response_with_structured_output(structured_response)
    
    return {
            'messages': [final_response], 
            'confidence': structured_response.confidence, 
            'verdict': structured_response.verdict, 
            'evidence': structured_response.evidence
        }

def audio_node(state: FactCheckState) -> FactCheckState:
    print("Processing audio input...")

    audio_file = open(state['audio_path'], "rb")

    translation = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file,
    )

    return {'claim': translation.text}

def store_record(state: FactCheckState) -> FactCheckState:
    print("Storing record...")

    store_claim_cache(
                        embedding=state['embedding'],
                        claim=state['claim'],
                        credibility=state['confidence'],
                        content=state['evidence']
                    )
    
    return {}