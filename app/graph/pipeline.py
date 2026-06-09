from app.graph.schema import WorkflowState
from app.agents.triage import TriageAgent
from app.agents.coder import DiagnosticCoderAgent
from app.agents.documenter import ClinicalDocumentationAgent
from app.agents.imaging import RadiologyAgent
from langgraph.graph import START, END, StateGraph


def assemble_pipeline():
    graph = StateGraph(WorkflowState)

    graph.add_node("triage", TriageAgent().execute)
    graph.add_node("coding", DiagnosticCoderAgent().execute)
    graph.add_node("documentation", ClinicalDocumentationAgent().execute)
    graph.add_node("imaging", RadiologyAgent().execute)

    graph.add_edge(START, "triage")

    graph.add_conditional_edges("triage", lambda s: s.task, {
        "coding": "coding",
        "documentation": "documentation",
        "imaging": "imaging",
    })

    graph.add_edge("coding", END)
    graph.add_edge("documentation", END)
    graph.add_edge("imaging", END)

    return graph.compile()
