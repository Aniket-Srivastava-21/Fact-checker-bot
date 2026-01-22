from langsmith import traceable
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.state import LLMResponse


load_dotenv()

# Using OpenAI Key
client = OpenAI()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""]
)

def chunk_text(text):
    return splitter.split_text(text)

@traceable(name="embed_text")
def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        dimensions=512,
    )

    return response.data[0].embedding



@traceable(name="get_final_response_with_structured_output")
def get_final_response_with_structured_output(structured_output: LLMResponse):
    return f"""
Verdict: {structured_output.verdict}
Confidence: {structured_output.confidence}
Evidence: {structured_output.evidence}
"""

@traceable(name="get_retrieved_data")
def get_retrieved_data(retrieved):
    output = ""

    for r in retrieved:
        p = r.payload
        output += f"""
                Content: {p['content']}
                Credibility: {p['credibility']}
                """

    return output

@traceable(name="llm_decide_reuse")
def llm_decide_reuse(new_claim, retrieved):
    prompt = f"""
                New claim:
                {new_claim}

                Already available data with corresponding credibility scores:

            """

    prompt += get_retrieved_data(retrieved)

    prompt += """
            Question:
            Can the new claim be fact-checked using the above past data alone?

            Answer only one word:
            YES or NO
            """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"LLM Reuse Decision: {resp.choices[0].message.content.strip()}")

    return resp.choices[0].message.content.strip() == "YES"