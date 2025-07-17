# Chunking Evaluation

This package, developed as part of our research detailed in the [Chroma Technical Report](https://research.trychroma.com/evaluating-chunking), provides tools for text chunking and evaluation. It allows users to compare different chunking methods and includes implementations of several novel chunking strategies.

## Features

- **Compare Chunking Methods**: Evaluate and compare various popular chunking strategies.
- **Novel Chunking Methods**: Implementations of new chunking methods such as `ClusterSemanticChunker` and `LLMChunker`.
- **Evaluation Framework**: Tools to generate domain-specific datasets and evaluate retrieval quality in the context of AI applications.

## Quick Start

You can immediately test the package via [Google Colab](https://colab.research.google.com/drive/1J5ALtDf0_RrswRz2fktjFVeFxe2jbXuJ?usp=sharing).

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/brandonstarxel/chunking_evaluation.git
```


# Evaluating Your Own Custom Chunker
This example shows how to implement your own chunking logic and evaluate its performance.
```python
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions

# Define a custom chunking class
class CustomChunker(BaseChunker):
    def split_text(self, text):
        # Custom chunking logic
        return [text[i:i+1200] for i in range(0, len(text), 1200)]

# Instantiate the custom chunker and evaluation
chunker = CustomChunker()
evaluation = GeneralEvaluation()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Evaluate the chunker
results = evaluation.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
# 'recall_mean': 0.8091207841640163, 'recall_std': 0.3792297991952294}
```

# Evaluating a Custom Embedding Function
```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings

# Instantiate instance of ef
default_ef = MyEmbeddingFunction()

# Evaluate the embedding function with a chunker
results = evaluation.run(chunker, default_ef)
```

# Usage and Evaluation of ClusterSemanticChunker
This example demonstrates how to use our ClusterSemanticChunker and how you can evaluate it yourself.
```python
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

# Instantiate evaluation
evaluation = GeneralEvaluation()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Instantiate chunker and run the evaluation
chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)
results = evaluation.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.18255175232840098, 'iou_std': 0.12773219595465307, 
# 'recall_mean': 0.8973469551927365, 'recall_std': 0.29042203879923994}
```

## Synthetic Dataset Pipeline for Domain Specific Evaluation

Here are the steps you can take to develop a sythetic dataset based off your own corpora for domain specific evaluation.

1. **Initialize the Environment**:

    ```python
    from chunking_evaluation import SyntheticEvaluation
    
    # Specify the corpora paths and output CSV file
    corpora_paths = [
        'path/to/chatlogs.txt',
        'path/to/finance.txt',
        # Add more corpora files as needed
    ]
    queries_csv_path = 'generated_queries_excerpts.csv'
    
    # Initialize the evaluation
    evaluation = SyntheticEvaluation(corpora_paths, queries_csv_path, openai_api_key="OPENAI_API_KEY")
    ```

2. **Generate Queries and Excerpts**:

    ```python
    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts()
    ```

3. **Apply Filters**:

    ```python
    # Apply filter to remove queries with poor excerpts
    evaluation.filter_poor_excerpts(threshold=0.36)
    
    # Apply filter to remove duplicates
    evaluation.filter_duplicates(threshold=0.6)
    ```

4. **Run the Evaluation**:

    ```python
    from chunking_evaluation import BaseChunker

    # Define a custom chunking class
    class CustomChunker(BaseChunker):
        def split_text(self, text):
            # Custom chunking logic
            return [text[i:i+1200] for i in range(0, len(text), 1200)]

    # Instantiate the custom chunker
    chunker = CustomChunker()

    # Run the evaluation on the filtered data
    results = evaluation.run(chunker)
    print("Evaluation Results:", results)
    ```

2. **Optional: If generation is unable to generate queries try approximate excerpts**

    ```python
    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts(approximate_excerpts=True)
    ```
## Package Dependancies:
The following will be installed along with the package:
- tiktoken
- fuzzywuzzy
- pandas
- numpy
- tqdm
- chromadb
- python-Levenshtein
- openai
- anthropic
- attrs

## Citation

If you use this package in your research, please cite our technical report:
```
@techreport{smith2024evaluating,
  title = {Evaluating Chunking Strategies for Retrieval},
  author = {Smith, Brandon and Troynikov, Anton},
  year = {2024},
  month = {July},
  institution = {Chroma},
  url = {https://research.trychroma.com/evaluating-chunking},
}
```

## Contributions
We welcome contributions and are excited you'd like to get involved! 
Make sure your pull request goes to the dev branch. We will test it and then later merge it to main.
