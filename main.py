import argparse
from src.graph import build_graph
from src.ingestion import ingest_wikipedia

def run_chat(claim: str, audio_path: str = None):
    bot = build_graph()
    
    initial_state = {
        'messages': [],
        'input_type': 'audio' if audio_path else 'text',
        'audio_path': audio_path,
        'claim': claim
    }
    
    print(f"Starting fact check for: {claim}")
    response = bot.invoke(initial_state)
    
    # Extract final answer from messages
    last_msg = response['messages'][-1]
    print("\n=== FINAL RESULT ===")
    print(last_msg.content)

def run_ingestion():
    topics = ["Japan", "Barack Obama", "Climate Change"]
    for topic in topics:
        ingest_wikipedia(topic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact Checking Bot")
    parser.add_argument("--mode", choices=["chat", "ingest"], default="chat", help="Run mode")
    parser.add_argument("--claim", type=str, help="Claim to verify (text mode)")
    parser.add_argument("--audio", type=str, help="Path to audio file (audio mode)")

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingestion()
    else:
        if not args.claim and not args.audio:
            # Default for testing
            run_chat("India gained independence in september 1947.")
        else:
            run_chat(args.claim, args.audio)