# Contextual RAG System Demonstration

This document demonstrates the setup and usage of the Contextual RAG system defined in `rag.py`. It covers:

1.  **Prerequisites:** Setting up the required environment (Docker, Ollama, Python packages).
2.  **Starting Services:** Launching Elasticsearch and ensuring Ollama is running.
3.  **Preparing Data:** Defining sample text to be indexed.
4.  **Initializing the RAG:** Creating instances of the database classes.
5.  **Indexing Data (First Time):** Processing, contextualizing, embedding, and indexing the sample text.
6.  **Loading Data (Subsequent Times):** Loading pre-processed data from disk.
7.  **Querying:** Asking questions to the RAG system.

## 1. Prerequisites

Before proceeding, ensure you have the following installed and configured:

*   **Docker:** To run Elasticsearch. Install from [Docker's official website](https://www.docker.com/get-started).
*   **Ollama:** For running local LLMs. Install from [Ollama's official website](https://ollama.com/).
*   **Python 3.8+:** With the necessary packages.
*   **Code Files:**
    *   `rag.py`: The file containing your RAG system code.
    *   `ollama_client.py`: The file containing your `ChatOllama` client class.
    *   Place both files in the same directory as where you intend to run your Python script or notebook, or ensure they are in your Python path.

**Install Python Packages:**
Open your terminal or command prompt and run:

```bash
pip install ollama elasticsearch==8.8.0 numpy pydantic tqdm nltk
```
*(Optional: If running interactively in Jupyter, also install `ipykernel jupyter`)*

**NLTK Data (Required by `ContextualVectorDB`):**
The `nltk.download()` calls are included in the `ContextualVectorDB` constructor in `rag.py`. They should run automatically the first time you initialize the `ContextualVectorDB` object in your Python environment.

## 2. Start Services

### Start Elasticsearch using Docker

Open a **separate terminal window** and run the following Docker command. This will download the Elasticsearch image (if you don't have it) and start a container named `elasticsearch`.

```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
```

**Note:** It might take a minute or two for Elasticsearch to fully initialize. You can check the logs using `docker logs elasticsearch`. Keep this terminal window open and the container running.

**(Optional) Stop and Remove Container Later:**
When you're finished, you can stop and remove the container using:
```bash
docker stop elasticsearch
docker rm elasticsearch
```

### Ensure Ollama is Running

Make sure your Ollama service is running locally. You can typically start it by:

1.  Running the Ollama application (if you installed the desktop version).
2.  Or, opening **another separate terminal** and running: `ollama serve`

Keep the Ollama service running while you use the RAG system.

**Pull Required Models:**
Your code uses several Ollama models. Ensure they are downloaded. Run these commands in your terminal:

```bash
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
ollama pull llama3.2:latest # Used for sub-question generation
ollama pull deepseek-r1:32b # Used for final answer generation
```

**Important:** The RAG system code needs to be able to connect to Ollama (usually at `http://localhost:11434`).

## 3. Prepare Input Data

Define the document text that you want to load into the RAG system. For this demo, we'll use a short article about the Apollo 11 mission. You can replace this `sample_article` variable with your own document content in your Python script.

```python
# In your Python script or interactive session:
sample_article = """
Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. Command module pilot Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface. Armstrong and Aldrin spent 21 hours, 36 minutes on the lunar surface at a site they named Tranquility Base before lifting off to rejoin Columbia in lunar orbit.

The mission was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, Florida, on July 16, 1969, at 13:32 UTC, and it was the fifth crewed mission of NASA's Apollo program. The Apollo spacecraft had three parts: a command module (CM) with a cabin for the three astronauts, the only part that returned to Earth; a service module (SM), which supported the command module with propulsion, electrical power, oxygen, and water; and a lunar module (LM) that had two stages – a descent stage for landing on the Moon and an ascent stage to place the astronauts back into lunar orbit.

After being sent to the Moon by the Saturn V's third stage, the astronauts separated the spacecraft from it and traveled for three days until they entered lunar orbit. Armstrong and Aldrin then moved into Eagle and landed in the Sea of Tranquility on July 20. The astronauts used Eagle's ascent stage to lift off from the lunar surface and rejoin Collins in the command module. They jettisoned Eagle before they performed the maneuvers that propelled Columbia out of the last of its 30 lunar orbits onto a trajectory back to Earth. They returned to Earth and splashed down in the Pacific Ocean on July 24, 1969, after more than eight days in space. The exact date of splashdown was July 24, 1969.

Armstrong's first step onto the lunar surface was broadcast on live TV to a worldwide audience. He described the event as "one small step for [a] man, one giant leap for mankind." Apollo 11 effectively ended the Space Race and fulfilled a national goal proposed in 1961 by President John F. Kennedy: "before this decade is out, of landing a man on the Moon and returning him safely to the Earth."
"""
```

## 4. Initialize the RAG System

Now, import the necessary classes from your `new_rag_local.py` file and create instances.

```python
# In your Python script or interactive session:
import time
import os
from rag import ContextualVectorDB, ContextualBM25VectorDB

# ollama_client.py containing ChatOllama is used internally by rag.py

# Give Elasticsearch a moment to start up if you just ran the docker command
print("Waiting for Elasticsearch to potentially start...")
time.sleep(15)  # Adjust sleep time if needed
print("Proceeding with initialization...")

# Initialize the Vector DB. The name ('Apollo11Demo') determines the subdirectory
# in './data/' where the .pkl file will be saved/loaded.
# The first time this runs, it will also download NLTK data if not present.
vector_db = ContextualVectorDB("Apollo11Demo")

# Initialize the combined BM25 + Vector DB system.
# This will attempt to connect to Elasticsearch (ensure it's running!)
# and create the index if it doesn't exist.
db_hybrid = ContextualBM25VectorDB(vector_db)

print("RAG System Initialized.")
print(f"Vector DB data path: {vector_db.db_path}")
print(f"Elasticsearch index name: {db_hybrid.es_bm25.index_name}")

# Define the path for the vector DB file based on the name
db_file_path = vector_db.db_path  # e.g., './data/Apollo11Demo/contextual_vector_db.pkl'
```

## 5. Indexing Data (First Time Use)

If this is the first time you are running the system with this specific `ContextualVectorDB` name (`Apollo11Demo`), you need to load and process the data. The `.pkl` file (e.g., `./data/Apollo11Demo/contextual_vector_db.pkl`) will not exist yet.

The `vector_db.load_data` method will:
1.  Chunk the input `document`.
2.  Generate a contextual summary for each chunk using Ollama (`llama3.2:1b`).
3.  Embed the chunk + summary using Ollama (`mxbai-embed-large`).
4.  Store embeddings and metadata in memory.
5.  Save embeddings and metadata to the `.pkl` file.

After `load_data` finishes, the metadata needs to be indexed into Elasticsearch by `ContextualBM25VectorDB`.

**Note:** This step involves calls to Ollama and can take time.

```python
# --- Run this code block only if the .pkl file doesn't exist ---

# Check if the database file already exists
db_file_exists = os.path.exists(db_file_path)

if not db_file_exists:
    print(f"Database file {db_file_path} not found. Running load_data...")
    # Specify the number of parallel threads for contextualization
    # Adjust based on your CPU cores and Ollama's ability to handle concurrent requests
    num_threads = 4 # Example value, adjust as needed

    start_time = time.time()
    # Pass the sample_article (or your document) to load_data
    vector_db.load_data(document=sample_article, parallel_threads=num_threads)
    end_time = time.time()

    print(f"\nData loading and vector DB creation finished in {end_time - start_time:.2f} seconds.")

    # After loading data, the vector_db has metadata. Index it into ES.
    print("Indexing data into Elasticsearch (BM25)...")
    # Use the already initialized db_hybrid instance which contains the es_bm25 component
    db_hybrid.es_bm25.index_documents(vector_db.metadata)
    print("Elasticsearch indexing complete.")
else:
    print(f"Database file {db_file_path} already exists. Skipping initial load_data.")
    print("Proceed to the 'Loading Data (Subsequent Times)' step if needed, or directly to querying.")

```

## 6. Loading Data (Subsequent Times)

If you have already run the `load_data` step once for this database name (`Apollo11Demo`), the `.pkl` file should exist. You can load the embeddings and metadata directly from the file, which is much faster than re-processing.

The Elasticsearch index should persist as long as the Docker container is running or its data volume persists. This code block also includes a check to ensure the ES index has documents and attempts re-indexing if it's empty but the `.pkl` file was loaded successfully.

```python
# --- Run this code block if the .pkl file already exists and you need to load it ---

# Check if the database file exists before trying to load
db_file_exists = os.path.exists(db_file_path)

# Ensure vector_db and db_hybrid are initialized as in Step 4
# (You might need to re-run Step 4 if in a new session)

if db_file_exists:
    # Only load if embeddings are not already in memory (e.g., starting a new script)
    if not vector_db.embeddings:
        print(f"Loading data from {db_file_path}...")
        try:
            vector_db.load_db() # Loads embeddings and metadata from pickle
            print("Data loaded successfully from file into vector_db.")

            # Optional: Verify Elasticsearch index has documents (assuming it persisted)
            # Force a refresh and check count
            try:
                print("Verifying Elasticsearch index...")
                db_hybrid.es_bm25.es_client.indices.refresh(index=db_hybrid.es_bm25.index_name)
                count = db_hybrid.es_bm25.es_client.count(index=db_hybrid.es_bm25.index_name)['count']
                print(f"Elasticsearch index '{db_hybrid.es_bm25.index_name}' contains {count} documents.")

                # If ES index is empty but we loaded metadata, try re-indexing
                if count == 0 and vector_db.metadata:
                     print("Elasticsearch index is empty. Re-indexing based on loaded metadata...")
                     db_hybrid.es_bm25.index_documents(vector_db.metadata)
                     print("Elasticsearch re-indexing complete.")
                elif count > 0 and count != len(vector_db.metadata) and vector_db.metadata:
                     print(f"Warning: Elasticsearch count ({count}) doesn't match loaded metadata count ({len(vector_db.metadata)}). Index might be stale.")

            except Exception as e:
                print(f"Could not verify Elasticsearch index count: {e}")
                print("Ensure Elasticsearch is running and accessible at http://localhost:9200.")

        except Exception as e:
            print(f"Error loading data from file: {e}")
            print("Ensure the .pkl file is not corrupted and was created successfully.")
    else:
        print("Vector DB data seems already loaded in memory.")
else:
    print(f"Database file {db_file_path} does not exist.")
    print("Please run the 'Indexing Data (First Time Use)' step first.")

```

## 7. Querying the RAG System

Now you can ask questions using the `answer_query_base` method of the `ContextualBM25VectorDB` instance (`db_hybrid`). Make sure Ollama and Elasticsearch are running.

This method:
1.  Generates sub-questions from your query using `llama3.2:latest`.
2.  Performs hybrid retrieval (semantic + BM25) for each sub-question.
3.  Aggregates the retrieved context.
4.  Generates the final answer using `deepseek-r1:32b`.

```python
# Ensure vector_db and db_hybrid are initialized and data is loaded (either via load_data or load_db)

# Example Queries:
query1 = "When did Apollo 11 land on the moon?"
query2 = "Who were the astronauts on the Apollo 11 mission?"
query3 = "What was the goal of the Apollo program mentioned by JFK?"
# Example query from user request - may not be answerable from the text:
query4 = "When was the paper published, provide to me the exact date?"

queries_to_run = [query1, query2, query3, query4]

for i, query in enumerate(queries_to_run):
    print(f"\n--- Query {i+1}: {query} ---")
    start_time = time.time()
    try:
        # Make sure db_hybrid is ready (initialized and data loaded/indexed)
        if vector_db.metadata: # Basic check if data seems loaded
             answer = db_hybrid.answer_query_base(query)
             end_time = time.time()
             print(f"Answer generated in {end_time - start_time:.2f} seconds.")
             print(f"\nAnswer:\n{answer}")
        else:
             print("Error: Cannot query. Vector DB metadata is empty. Load or index data first.")

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        # Add more specific error handling if needed, e.g., for connection errors
        import traceback
        traceback.print_exc()

    print("-" * (len(query) + 14)) # Separator

```

## 8. Conclusion

This document outlined the workflow for setting up and using the Contextual RAG system:

*   Setting up dependencies (Docker for Elasticsearch, Ollama).
*   Starting the required services.
*   Loading and processing source documents: chunking, contextualization (Ollama), embedding (Ollama), and indexing (Vector DB + Elasticsearch BM25).
*   Loading pre-processed data for faster startup on subsequent runs.
*   Performing hybrid retrieval using sub-questions and generating answers with Ollama.

Remember to stop the background services when you are finished if they are no longer needed.

**Stop Elasticsearch (in its terminal or a new one):**
```bash
docker stop elasticsearch
docker rm elasticsearch
```

**Stop Ollama:**
*   Quit the Ollama application.
*   Or, if you ran `ollama serve` in a terminal, press `Ctrl+C` in that terminal.
