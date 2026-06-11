# Dataset Recommender

A machine learning project that recommends relevant datasets based on user queries using TF-IDF similarity and metadata collected from Hugging Face datasets.

## Overview

Finding the right dataset is often a time-consuming task for data scientists and machine learning practitioners. This project aims to simplify dataset discovery by building a recommendation engine that returns datasets relevant to a user's interests, domain, or task.

The current implementation uses TF-IDF vectorization on dataset metadata and descriptions to identify similar datasets and generate recommendations.

---

## Features

### Stage 1 (Completed)

- Dataset collection from Hugging Face
- Data cleaning and preprocessing
- Unified dataset catalog creation
- Data quality analysis
- Exploratory visualizations
- TF-IDF recommendation engine
- Example recommendation reports

### Planned Features

- Semantic search using embeddings
- Sentence Transformers integration
- Popularity-aware ranking
- Dataset filtering by:
  - Domain
  - Task
  - Language
  - License
- Interactive Streamlit web application
- Hybrid recommendation system
- API deployment

---

## Project Structure

```text
dataset_recommender/
│
├── data/
│   ├── raw/
│   │   └── huggingface_datasets.csv
│   │
│   └── processed/
│       └── unified_catalog.csv
│
├── notebooks/
│   └── data_quality_analysis.ipynb
│
├── reports/
│   ├── figures/
│   ├── recommender_examples/
│   └── stage1_summary.md
│
├── src/
│   ├── scraping/
│   │   └── scrape_huggingface.py
│   │
│   ├── preprocessing/
│   │   ├── inspect_hf_data.py
│   │   ├── build_unified_catalog.py
│   │   ├── catalog_statistics.py
│   │   └── data_quality_analysis.py
│   │
│   └── recommender/
│       └── tfidf_recommender.py
│
├── requirements.txt
└── README.md
```

---

## Dataset Source

The project currently uses publicly available datasets from Hugging Face.

Dataset metadata includes:

- Dataset name
- Description
- Tags
- Task information
- Additional metadata

---

## Methodology

### 1. Data Collection

Dataset metadata is collected from Hugging Face and stored locally.

### 2. Data Processing

The collected data is:

- Cleaned
- Standardized
- Merged into a unified catalog

### 3. Feature Engineering

Dataset descriptions and metadata are combined into a text representation used for similarity computation.

### 4. TF-IDF Vectorization

TF-IDF converts textual metadata into numerical feature vectors.

### 5. Similarity Search

Cosine similarity is used to identify datasets that are most similar to a user query or reference dataset.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/IqraIshfaq14/dataset_recommender.git
cd dataset_recommender
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Recommender

```bash
python src/recommender/tfidf_recommender.py
```

The recommender returns the most relevant datasets based on textual similarity.

---

## Example Queries

### Query: Iris Dataset

Returns similar:

- Classification datasets
- Tabular datasets
- Benchmark datasets

### Query: Image Classification

Returns similar:

- Vision datasets
- Object detection datasets
- Image segmentation datasets

### Query: Mathematical Reasoning

Returns similar:

- GSM8K
- Reasoning benchmarks
- Educational datasets

---

## Data Quality Analysis

The project includes exploratory analysis and visualizations such as:

- Missing value analysis
- Dataset description length distribution
- Tag frequency analysis
- Catalog statistics

Generated figures are stored in:

```text
reports/figures/
```

---

## Current Results

Stage 1 successfully demonstrates:

- Automated dataset collection
- Dataset catalog construction
- Metadata exploration
- TF-IDF recommendation system

This serves as a strong baseline for future semantic recommendation approaches.

---

## Future Roadmap

### Stage 2

- Improve recommendation quality
- Increase recommendation count
- Add metadata filters
- Improve ranking strategy

### Stage 3

- Sentence Transformer embeddings
- Semantic search
- Hybrid recommendation system

### Stage 4

- Streamlit application
- Interactive user interface
- Dataset search dashboard

### Stage 5

- Deployment
- REST API
- Cloud hosting

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Hugging Face Datasets
- Git
- GitHub

---

## Author

**Iqra Ishfaq**

GitHub: https://github.com/IqraIshfaq14

---

## License

This project is intended for educational, research, and portfolio purposes.
