import requests
import zipfile
import io
import os
import json
from tqdm import tqdm
from langchain.schema.document import Document
import base64

DATASET_URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip"
DATA_DIR = "data"


def download_and_unzip_data():
    """Downloads and unzips the Synthea dataset if not already present."""
    if os.path.exists(DATA_DIR):
        print("Dataset already downloaded.")
        return

    print("Downloading Synthea latest sample dataset...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print("Unzipping dataset...")
        z.extractall(DATA_DIR)
    print("Dataset ready.")


def load_and_curate_documents():
    """Loads data from the unzipped files and curates it into patient documents."""
    download_and_unzip_data()

    patient_notes = {}

    print("Parsing clinical notes from individual patient FHIR files...")

    all_json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    patient_files = [f for f in all_json_files if "_" in f]

    for filename in tqdm(patient_files, desc="Processing patient files"):
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)

                # First, find the main Patient ID for this file's bundle
                current_patient_id = None
                for entry in data.get("entry", []):
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") == "Patient":
                        current_patient_id = resource.get("id")
                        if current_patient_id:
                            break  # Found the patient ID, no need to look further

                if not current_patient_id:
                    continue  # Skip if no patient found in this file

                if current_patient_id not in patient_notes:
                    patient_notes[current_patient_id] = []

                # Now, find all DiagnosticReports and extract their narrative text
                for entry in data.get("entry", []):
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") == "DiagnosticReport":
                        if "presentedForm" in resource:
                            for form in resource["presentedForm"]:
                                if (
                                    "data" in form
                                    and form.get("contentType")
                                    == "text/plain; charset=utf-8"
                                ):
                                    try:
                                        # Decode the Base64 data to get the clinical note
                                        decoded_bytes = base64.b64decode(form["data"])
                                        note_text = decoded_bytes.decode("utf-8")
                                        if note_text:
                                            patient_notes[current_patient_id].append(
                                                note_text
                                            )
                                    except Exception as e:
                                        # This helps debug if a specific note fails to decode
                                        print(
                                            f"Warning: Could not decode base64 data in {filename}. Error: {e}"
                                        )
                                        pass
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file {filename}. Skipping.")
                continue

    # Aggregate the collected notes into LangChain Document objects
    docs = []
    for patient_id, notes in patient_notes.items():
        if notes:  # Only create a document if we successfully extracted notes
            # Join all notes for a single patient into one large document
            full_text = "\n\n--- Next Clinical Note ---\n\n".join(notes)
            doc = Document(page_content=full_text, metadata={"patient_id": patient_id})
            docs.append(doc)

    if not docs:
        print("\nCRITICAL ERROR: No documents were processed.")
        print("Please check the following:")
        print(f"1. The directory '{DATA_DIR}' exists.")
        print("2. It contains patient JSON files (e.g., 'Abdul218_Nienow652...json').")
        print(
            "3. The JSON files contain 'DiagnosticReport' resources with base64 data."
        )
    else:
        print(f"\nSuccessfully curated and processed data for {len(docs)} patients.")

    return docs
