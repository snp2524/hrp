from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_core.runnables import Runnable
from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node as LangChainNode,
    Relationship as LangChainRelationship,
)
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize connection to Neo4j
graph = Neo4jGraph()


# Schema for the relationships
class SimpleRelationship(BaseModel):
    """A simple, flat relationship between two entities."""

    start_node_name: str = Field(..., description="The name of the source entity.")
    start_node_type: str = Field(
        ..., description="The type of the source entity (e.g., 'Patient', 'Condition')."
    )
    end_node_name: str = Field(..., description="The name of the target entity.")
    end_node_type: str = Field(
        ...,
        description="The type of the target entity (e.g., 'Diagnosis', 'Treatment').",
    )
    relationship_type: str = Field(
        ..., description="The type of the relationship (e.g., 'HAS_DIAGNOSIS')."
    )


class GraphOutput(BaseModel):
    """A list of simple, flat relationships."""

    relationships: List[SimpleRelationship]


# Schema for the graph
graph_prompt_template = """
You are an expert medical archivist. Your task is to extract structured information from a patient's clinical notes.
From the text below, extract entities and their relationships.
The entire response must be a single JSON object with one key, "relationships", which contains a list of flat relationship objects.

Use the following entity types: 'Patient', 'Condition', 'Diagnosis', 'Treatment', 'Symptom'.
Use the following relationship types: 'HAS_CONDITION', 'HAS_DIAGNOSIS', 'RECEIVES_TREATMENT', 'EXHIBITS_SYMPTOM'.

Example Response Format:
{{
  "relationships": [
    {{
      "start_node_name": "John Doe",
      "start_node_type": "Patient",
      "end_node_name": "Viral sinusitis",
      "end_node_type": "Diagnosis",
      "relationship_type": "HAS_DIAGNOSIS"
    }},
    {{
      "start_node_name": "John Doe",
      "start_node_type": "Patient",
      "end_node_name": "Amoxicillin",
      "end_node_type": "Treatment",
      "relationship_type": "RECEIVES_TREATMENT"
    }}
  ]
}}

Text to process:
{text}
"""


def extract_graph_from_text(
    llm: Runnable, text: str, patient_id: str
) -> List[GraphDocument]:
    """Uses an LLM to extract graph information from text."""
    structured_llm = llm.with_structured_output(GraphOutput)
    chain = PromptTemplate.from_template(graph_prompt_template) | structured_llm

    try:
        results = chain.invoke({"text": text})

        graph_docs = []
        for rel in results.relationships:
            # Create nodes with proper IDs and types
            source_node = LangChainNode(
                id=rel.start_node_name,
                type=rel.start_node_type,
                properties={},
            )
            target_node = LangChainNode(
                id=rel.end_node_name,
                type=rel.end_node_type,
                properties={},
            )

            # Create relationship with proper source/target references
            relationship = LangChainRelationship(
                source=source_node,
                target=target_node,
                type=rel.relationship_type,
                properties={},
            )

            # Create a Document object for the source field
            from langchain.schema.document import Document

            source_doc = Document(
                page_content=text[:200] + "..." if len(text) > 200 else text,
                metadata={"patient_id": patient_id},
            )

            # Create GraphDocument with proper Document source
            graph_doc = GraphDocument(
                nodes=[source_node, target_node],
                relationships=[relationship],
                source=source_doc,
            )
            graph_docs.append(graph_doc)

        return graph_docs
    except Exception as e:
        print(f"Error extracting graph from text for patient {patient_id}: {e}")
        return []


def populate_knowledge_graph(llm, documents, max_workers: int = 5):
    """Populates the Neo4j database from a list of documents using multithreading.

    Args:
        llm: The language model (Runnable) used for extraction.
        documents: List of LangChain documents to process.
        max_workers: Number of parallel threads to use for extraction. Adjust based on
            your model/provider rate-limits and machine capabilities.
    """
    print("Populating Knowledge Graph... This may take a while.")

    # Clear existing data
    try:
        graph.query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"Warning: Could not clear existing graph data: {e}")
        print("This might be because Neo4j is not running. Continuing anyway...")

    successful_extractions = 0
    total_relationships = 0

    from tqdm import tqdm

    print(
        f"\nProcessing {len(documents)} documents with up to {max_workers} workers..."
    )

    def _process_doc(idx, doc):
        """Helper to extract relationships for a single document."""
        patient_id = doc.metadata.get("patient_id", "Unknown")
        graph_docs = extract_graph_from_text(llm, doc.page_content, patient_id)
        return idx, patient_id, graph_docs

    # Launch extraction tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_doc, i, doc) for i, doc in enumerate(documents)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting relationships",
            unit="doc",
        ):
            try:
                idx, patient_id, graph_documents = future.result()
            except Exception as e:
                tqdm.write(f"  Extraction failed in worker: {e}")
                continue

            tqdm.write(
                f"Processing document {idx + 1}/{len(documents)} for patient {patient_id}"
            )

            if graph_documents:
                successful_extractions += 1
                relationships_in_doc = sum(
                    len(gd.relationships) for gd in graph_documents
                )
                total_relationships += relationships_in_doc

                try:
                    graph.add_graph_documents(graph_documents)
                    tqdm.write(f"  Added {relationships_in_doc} relationships to graph")
                except Exception as e:
                    tqdm.write(f"  Failed to add to graph: {e}")

    print(
        f"\nKnowledge Graph population complete. Successfully extracted relations from {successful_extractions}/{len(documents)} documents."
    )
    print(f"Total relationships extracted: {total_relationships}")

    # Try to show schema from Neo4j
    print("Final Graph Schema:")
    print(graph.schema)


def create_graph_rag_chain(llm):
    """Pipeline 4: Knowledge Graph RAG."""
    print("Building Graph RAG chain...")

    try:
        chain = GraphCypherQAChain.from_llm(
            graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
        )
        return chain
    except Exception as e:
        print(f"Warning: Could not create Graph RAG chain: {e}")
        print("This might be because Neo4j is not running.")
        # Return a simple fallback chain that just uses the LLM directly
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            "You are a medical assistant. Answer the following question based on your knowledge: {query}"
        )
        return LLMChain(llm=llm, prompt=prompt)
