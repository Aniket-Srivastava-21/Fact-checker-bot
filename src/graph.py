from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal

from src.state import FactCheckState
from src.tools import tools
from src.nodes import (
    verify_claim, audio_node, embed_node, cache_lookup, 
    store_record, reuse_verdict_node, formatting_node
)

def check_input_type(state: FactCheckState) -> Literal['embed_node', 'audio_node']:
    print(f"Input type is: {state['input_type']}")

    if state['input_type'] == 'text':
        return 'embed_node'
    elif state['input_type'] == 'audio':
        return 'audio_node'
    else:
        raise ValueError("Invalid input type")
    
def cache_router(state) -> Literal['reuse_verdict_node', 'verify_claim']:
    return "reuse_verdict_node" if state["cache_hit"] else "verify_claim"

def custom_router(state: FactCheckState):
    # Check if the LLM called a tool
    route = tools_condition(state)
    
    # tools_condition returns "__end__" if no tools were called
    if route == "__end__":
        return "formatting_node"
    
    # Otherwise, it returns "tools"
    return route

def build_graph():
    # Create the StateGraph
    graph = StateGraph(FactCheckState)


    graph.add_node('verify_claim', verify_claim)
    graph.add_node('audio_node', audio_node)
    graph.add_node('embed_node', embed_node)
    graph.add_node('cache_lookup', cache_lookup)
    graph.add_node('store_record', store_record)
    graph.add_node('reuse_verdict_node', reuse_verdict_node)
    graph.add_node("formatting_node", formatting_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_conditional_edges(START, check_input_type)
    graph.add_edge('audio_node', 'embed_node')
    graph.add_edge('embed_node', 'cache_lookup')
    graph.add_conditional_edges('cache_lookup', cache_router)
    graph.add_conditional_edges(
        'verify_claim', 
        custom_router, 
        {
            "tools": "tools", 
            "formatting_node": "formatting_node"
        }
    )
    graph.add_edge("tools", 'verify_claim')
    graph.add_edge("formatting_node", "store_record")
    graph.add_edge('reuse_verdict_node', END)
    graph.add_edge('store_record', END)

    chatbot = graph.compile()

    return chatbot