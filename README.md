# Chroma Research

A package for text chunking and benchmarking.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/brandonstarxel/chroma_research.git
```


# Benchmarking with Your Own Custom Chunker
This example shows how to implement your own chunking logic and benchmark its performance.
```python
from chroma_research import BaseChunker, GeneralBenchmark
from chromadb.utils import embedding_functions

# Define a custom chunking class
class CustomChunker(BaseChunker):
    def split_text(self, text):
        # Custom chunking logic
        return [text[i:i+1200] for i in range(0, len(text), 1200)]

# Instantiate the custom chunker and benchmark
chunker = CustomChunker()
benchmark = GeneralBenchmark()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Run the benchmark
results = benchmark.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
# 'recall_mean': 0.8091207841640163, 'recall_std': 0.3792297991952294}
```

# Usage and Benchmarking of ClusterSemanticChunker
This example demonstrates how to use our ClusterSemanticChunker and how you can benchmark it yourself.
```python
from chroma_research import BaseChunker, GeneralBenchmark
from chroma_research.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

# Instantiate benchmark
benchmark = GeneralBenchmark()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Instantiate chunker and run the benchmark
chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)
results = benchmark.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.18255175232840098, 'iou_std': 0.12773219595465307, 
# 'recall_mean': 0.8973469551927365, 'recall_std': 0.29042203879923994}
```

## Synthetic Benchmark Pipeline

Here are the steps you can take to develop a domain specific benchmark based off your own corpora.

1. **Initialize the Environment**:

    ```python
    from chroma_research import SyntheticBenchmark
    
    # Specify the corpora paths and output CSV file
    corpora_paths = [
        'path/to/chatlogs.txt',
        'path/to/finance.txt',
        # Add more corpora files as needed
    ]
    questions_csv_path = 'generated_questions_highlights.csv'
    
    # Initialize the benchmark
    benchmark = SyntheticBenchmark(corpora_paths, questions_csv_path, openai_api_key="OPENAI_API_KEY")
    ```

2. **Generate Questions and Highlights**:

    ```python
    # Generate questions and highlights, and save to CSV
    benchmark.generate_questions_and_highlights()
    ```

3. **Apply Filters**:

    ```python
    # Apply filter to remove questions with poor highlights
    benchmark.filter_poor_highlights(threshold=0.36)
    
    # Apply filter to remove duplicates
    benchmark.filter_duplicates(threshold=0.6)
    ```

4. **Run the Benchmark**:

    ```python
    from chroma_research import BaseChunker

    # Define a custom chunking class
    class CustomChunker(BaseChunker):
        def split_text(self, text):
            # Custom chunking logic
            return [text[i:i+1200] for i in range(0, len(text), 1200)]

    # Instantiate the custom chunker
    chunker = CustomChunker()

    # Run the benchmark on the filtered data
    results = benchmark.run(chunker)
    print("Benchmark Results:", results)
    ```

2. **Optional: If generation is unable to generate questions try approximate highlights**

    ```python
    # Generate questions and highlights, and save to CSV
    benchmark.generate_questions_and_highlights(approximate_highlights=True)
    ```
## Package Dependancies:
- chroma
- fuzzywuzzy
- pandas
- numpy
- tiktoken