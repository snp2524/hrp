import time
from config import get_gemini_llm, USE_GEMINI_EMBEDDINGS
from data_loader import load_and_curate_documents
from rag_pipelines import (
    get_local_embeddings,
    get_gemini_embeddings,
    create_standard_rag_chain,
    create_parent_document_rag_chain,
    create_sentence_window_rag_chain,
)
from knowledge_graph import populate_knowledge_graph, create_graph_rag_chain
from analysis import analyze_and_visualize_results


def run_experiment():
    """Orchestrates the entire experiment from data loading to results."""

    # 1. Load and Prepare Data
    print("--- Step 1: Loading Data ---")
    documents = load_and_curate_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # 2. Initialize Models
    print("\n--- Step 2: Initializing Models ---")
    llm = get_gemini_llm()
    if USE_GEMINI_EMBEDDINGS:
        print("--- Using Google Gemini Embeddings for this run ---")
        embeddings = get_gemini_embeddings()
    else:
        print("--- Using Local HuggingFace Embeddings for this run ---")
        embeddings = get_local_embeddings()

    # 3. Populate Knowledge Graph (One-time setup)
    # This step can be slow and costly. You can comment it out on subsequent runs
    # if the graph is already populated in your Neo4j instance.
    print("\n--- Step 3: Populating Knowledge Graph ---")
    populate_knowledge_graph(llm, documents, max_workers=10)

    # 4. Build RAG Pipelines
    print("\n--- Step 4: Building RAG Pipelines ---")
    pipelines = {
        "Standard RAG": create_standard_rag_chain(documents, llm, embeddings),
        "Parent Document (Late Chunking)": create_parent_document_rag_chain(
            documents, llm, embeddings
        ),
        "Sentence Window (Simulated)": create_sentence_window_rag_chain(
            documents, llm, embeddings
        ),
        "Knowledge Graph RAG": create_graph_rag_chain(llm),
    }

    # 5. Define Questions and Run Comparison
    print("\n--- Step 5: Running Comparison ---")

    # Enhanced test questions based on Synthea dataset characteristics
    questions = [
        # Vector search questions (semantic similarity)
        "What are the common symptoms and conditions associated with elderly patients?",
        "Describe the typical treatment patterns for patients with cardiovascular conditions.",
        "What are the most frequent medications prescribed to patients with chronic pain?",
        # Graph search questions (structured relationships)
        "Which patients were diagnosed with 'Viral sinusitis' and what treatments did they receive?",
        "Find patients with fractures and list all the procedures and medications they received.",
        "Show me patients with diabetes and their complete treatment history including medications and procedures.",
        # Hybrid questions (combining semantic and structured search)
        "What are the common health issues and treatments for patients who experienced domestic violence?",
        "Analyze the health progression of patients with gingival disease from diagnosis to treatment.",
        "Find patients with chronic conditions and describe their medication adherence patterns.",
        # Complex analytical questions
        "What is the relationship between patient socioeconomic status and access to certain treatments?",
        "How do treatment patterns differ between patients with and without insurance?",
        "What are the most common comorbidities in patients with mental health conditions?",
    ]

    # Store results for analysis
    results_data = {
        "questions": questions,
        "pipelines": list(pipelines.keys()),
        "answers": {},
        "timing": {},
        "quality_scores": {},
    }

    for i, question in enumerate(questions):
        print(f"\n{'=' * 20} Question {i + 1}: {question} {'=' * 20}")
        for name, pipeline in pipelines.items():
            print(f"\n--- Testing Pipeline: {name} ---")
            start_time = time.time()
            try:
                # The graph chain expects a 'query' key in the input dictionary
                if name == "Knowledge Graph RAG":
                    result = pipeline.invoke({"query": question})["result"]
                else:
                    result = pipeline.run(question)

                print(f"Answer: {result}")

                # Store results for analysis
                if name not in results_data["answers"]:
                    results_data["answers"][name] = {}
                    results_data["timing"][name] = {}
                    results_data["quality_scores"][name] = {}

                results_data["answers"][name][i] = result
                results_data["timing"][name][i] = time.time() - start_time

                # Simple quality scoring (length + keyword presence)
                quality_score = len(result) * 0.1
                if any(
                    keyword in result.lower()
                    for keyword in ["patient", "treatment", "diagnosis", "medication"]
                ):
                    quality_score += 20
                if result != "I don't know the answer.":
                    quality_score += 30
                results_data["quality_scores"][name][i] = quality_score

            except Exception as e:
                print(f"An error occurred: {e}")
                results_data["answers"][name][i] = f"Error: {e}"
                results_data["timing"][name][i] = 0
                results_data["quality_scores"][name][i] = 0

            end_time = time.time()
            print(f"Time Taken: {end_time - start_time:.2f} seconds")

    # 6. Results Analysis and Visualization
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS AND COMPARISON")
    print("=" * 80)

    analyze_and_visualize_results(results_data)


if __name__ == "__main__":
    run_experiment()
