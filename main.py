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

def run_ingestion(topics: list):
    """Ingests a list of Wikipedia topics into the vector database."""
    if not topics:
        # Default fallback list if nothing is provided
        topics = ["List of common misconceptions about history", "List of common misconceptions about science"]
        print(f"No topics provided. Using default list: {topics}")
    
    for topic in topics:
        print(f"\n--- Starting Ingestion: {topic} ---")
        ingest_wikipedia(topic)
    print("\n--- Ingestion Task Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact Checking Bot")
    
    # Existing modes
    parser.add_argument("--mode", choices=["chat", "ingest"], default="chat", 
                        help="Run mode: 'chat' to verify a claim, 'ingest' to add data to knowledge base")
    
    # Arguments for 'chat' mode
    parser.add_argument("--claim", type=str, help="Claim to verify (text mode)")
    parser.add_argument("--audio", type=str, help="Path to audio file (audio mode)")
    
    # New Argument for 'ingest' mode
    # nargs='+' allows passing multiple space-separated strings
    parser.add_argument("--topics", type=str, nargs='+', 
                        help="Wikipedia topics to ingest (e.g., --topics 'Elon Musk' 'Mars' 'Python programming')")

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingestion(args.topics)
    else:
        if not args.claim and not args.audio:
            # Interactive fallback if run without arguments
            default_claim = "India gained independence in september 1947."
            run_chat(default_claim)
        else:
            run_chat(args.claim, args.audio)