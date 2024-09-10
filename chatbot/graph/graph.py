from langgraph.graph import END, StateGraph, START

from state.state import GraphState
from nodes.nodes import (
    route,
    neo4j_user_node,
    neo4j_common_node,
    generate,
    summarize,
    update_knowledge_graph,
    route_summarization_usernode,
    video_processing,
    review_generation,
    regenerate,
    human_in_the_loop,
)


def graph():
    workflow = StateGraph(GraphState)

    # Define the nodes

    workflow.add_node("update_knowledge_graph", update_knowledge_graph)  # pdf-email
    workflow.add_node("neo4j_user_node", neo4j_user_node)  # neo4j_user_node
    workflow.add_node("neo4j_common_node", neo4j_common_node)  # neo4j_common_node
    workflow.add_node("generate", generate)
    workflow.add_node("summarize", summarize)
    workflow.add_node("route_summarization_usernode", route_summarization_usernode)
    # workflow.add_node("human_in_the_loop", human_in_the_loop)
    # workflow.add_node("send_email", send_email)  # send_email
    # workflow.add_node("review_generation", review_generation)  # review_generation
    # workflow.add_node("regenerate", regenerate)  # regenerate

    # Build graph
    workflow.add_conditional_edges(
        START,
        route,
        {
            "common_node": "neo4j_common_node",
            "user_node": "neo4j_user_node",
            "update_knowledge_graph": "update_knowledge_graph",
            "video_processing": "video_processing",
        },
    )
    workflow.add_edge("neo4j_common_node", "generate")
    workflow.add_edge("neo4j_user_node", "generate")
    workflow.add_edge("update_knowledge_graph", "route_summarization_usernode")

    workflow.add_conditional_edges(
        "route_summarization_usernode",
        lambda x: x["next"],
        {"neo4j_user_node": "neo4j_user_node", "summarize": "summarize"},
    )

    workflow.add_edge("summarize", "generate")
    # workflow.add_edge("generate", "review_generation")

    workflow.set_finish_point("generate")

    # workflow.add_conditional_edges(
    #     "review_generation",
    #     review_generation,
    #     {
    #         "transform_query": "transform_query",
    #         "proceed": human_in_the_loop,
    #     },
    # )

    # workflow.add_conditional_edges(
    #     "humain_in_the_loop",
    #     human_in_the_loop,
    #     {
    #         "email": "send_email",
    #         "ticket": "send_ticket",
    #         "respond": END,
    #     },
    # )

    # workflow.add_edge("send_email", END)

    # Compile
    graph = workflow.compile()

    return graph
