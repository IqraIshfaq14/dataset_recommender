# Dataset Recommendation System

## Project Overview

This project develops an intelligent dataset recommendation system that helps users discover relevant machine learning datasets based on metadata similarity and semantic understanding.

The system collects metadata from multiple dataset repositories, performs preprocessing and quality analysis, generates semantic embeddings, and recommends similar datasets using both traditional and modern recommendation techniques.

---

## Objectives

* Build a unified dataset catalog from multiple repositories.
* Analyze dataset metadata quality.
* Implement a TF-IDF based recommender system.
* Implement a semantic embedding-based recommender system.
* Compare recommendation approaches.
* Evaluate recommendation performance using ranking metrics.
* Deploy the system through an interactive web interface.

---

# Project Architecture

```text
Dataset Repositories
(Hugging Face, UCI, Kaggle)
            │
            ▼
    Metadata Collection
            │
            ▼
      Data Cleaning
            │
            ▼
     Unified Catalog
            │
            ├─────────────► TF-IDF Recommender
            │
            └─────────────► Embedding Generator
                                   │
                                   ▼
                      Semantic Embedding Recommender
                                   │
                                   ▼
                            Evaluation
                                   │
                                   ▼
                             Streamlit UI
```

---

# Technologies Used

| Category                    | Libraries / Tools                     |
| --------------------------- | ------------------------------------- |
| Data Collection             | Pandas, Hugging Face Hub, UCI ML Repo |
| Data Processing             | Pandas, NumPy                         |
| Text Representation         | TF-IDF, Sentence Transformers         |
| Machine Learning            | Scikit-learn                          |
| Embedding Models            | all-MiniLM-L6-v2                      |
| Evaluation                  | Precision@K, Recall@K, NDCG           |
| Experiment Tracking         | Weights & Biases (W&B)                |
| Hyperparameter Optimization | Optuna                                |
| Frontend                    | Streamlit                             |
| Deployment                  | Docker, Kubernetes                    |

---

# Stage 1: Data Collection

## Goal

Create a unified dataset catalog from multiple repositories.

## Sources

### Hugging Face

* Dataset ID
* Description
* Tags
* Downloads
* Likes
* URL

### UCI Repository

* Dataset Name
* Abstract
* Number of Instances
* Number of Features

### Kaggle (Planned)

* Dataset Metadata
* Tags
* Popularity Metrics

---

## Unified Schema

| Column      | Description                  |
| ----------- | ---------------------------- |
| dataset_id  | Unique dataset identifier    |
| title       | Dataset name                 |
| description | Dataset summary              |
| tags        | Dataset tags                 |
| task        | Machine learning task        |
| source      | Repository source            |
| downloads   | Popularity metric            |
| likes       | User feedback metric         |
| url         | Dataset URL                  |
| search_text | Combined searchable metadata |

---

# Stage 2: Data Preprocessing

## Data Cleaning

* Missing value handling
* Duplicate removal
* Text normalization
* Metadata standardization

## Tag Cleaning

Removed noisy tags such as:

```text
library:datasets
library:pandas
region:us
format:csv
```

Retained useful tags:

```text
language:en
modality:text
license:apache-2.0
```

---

# Stage 3: TF-IDF Recommender

## Method

Content-based recommendation using TF-IDF vectorization.

### Workflow

```text
Dataset Metadata
       ↓
TF-IDF Vectorizer
       ↓
Sparse Feature Vectors
       ↓
Cosine Similarity
       ↓
Recommendations
```

## Libraries

* Scikit-learn
* Pandas

## Techniques

* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Filtering

---

# Stage 4: Embedding Generation

## Method

Semantic embeddings generated using Sentence Transformers.

### Model

```text
all-MiniLM-L6-v2
```

### Workflow

```text
Title + Description + Tags
              ↓
Sentence Transformer
              ↓
384-Dimensional Embedding
              ↓
hf_embeddings.npy
```

### Output

```text
(10000, 384)
```

Each dataset is represented as a 384-dimensional semantic vector.

---

# Stage 5: Embedding Recommender

## Method

Semantic similarity search using vector embeddings.

### Workflow

```text
User Query
      ↓
Embedding Model
      ↓
Query Embedding
      ↓
Cosine Similarity
      ↓
Top-N Recommendations
```

### Advantages

* Understands semantic meaning
* Handles vocabulary differences
* More intelligent than keyword matching

---

# Stage 6: Evaluation

## Metrics

### Precision@K

Measures how many recommended datasets are relevant.

### Recall@K

Measures how many relevant datasets are successfully retrieved.

### NDCG

Measures ranking quality and recommendation ordering.

---

# Stage 7: Hyperparameter Optimization

## Tool

Optuna

## Purpose

Automatically search for optimal parameters including:

* TF-IDF feature size
* Similarity thresholds
* Recommendation ranking parameters

---

# Stage 8: Perturbation Analysis

## Goal

Evaluate recommender robustness.

### Experiments

* Remove tags
* Remove descriptions
* Corrupt metadata
* Measure recommendation degradation

This helps assess how sensitive the recommender is to incomplete metadata.

---

# Stage 9: Experiment Tracking

## Tool

Weights & Biases (W&B)

Tracked information:

* Model configurations
* Evaluation metrics
* Training experiments
* Hyperparameter searches

---

# Stage 10: Streamlit Frontend

## Features

* Search datasets by keyword
* Dataset recommendations
* Similarity scores
* Dataset metadata display

Example query:

```text
image classification datasets
```

---

# Expected Contributions

* Unified multi-source dataset catalog
* Content-based recommendation system
* Semantic embedding recommendation system
* Recommendation quality evaluation framework
* Robustness analysis through perturbation testing
* Interactive dataset discovery platform

---

# Future Work

* Hybrid recommender combining TF-IDF and embeddings
* User feedback integration
* Real-time dataset updates
* Large-scale deployment using Kubernetes

---

# Author

Master's Project – Dataset Recommendation System

Research Areas:

* Information Retrieval
* Recommender Systems
* Natural Language Processing
* Semantic Search
* Machine Learning
