# Ramayana-Data Text Analysis

## 1. Data Scraping & Preparation (`scrape_data.ipynb`)
### Source

The data is scraped from: `https://www.valmikiramayan.net/`

### Data Scraped

The script extracts shlokas and their corresponding English translations for **all Kandas** (Books) available on the site:

1.  Bala Kanda
2.  Ayodhya Kanda
3.  Aranya Kanda
4.  Kishkindha Kanda
5.  Sundara Kanda
6.  Yuddha Kanda

### Dataset 
[Ramayana Shloka Dataset](https://www.kaggle.com/datasets/parthpatil256/ramayana-shloka-dataset)

## Core NLP Techniques Used

1.  **Sentence Embeddings**: `sentence-transformers` (specifically `all-MiniLM-L6-v2`) are used to convert text (verses and statements) into dense vector representations. This allows for semantic similarity comparisons, crucial for retrieving relevant context.
2.  **Dimensionality Reduction**: Principal Component Analysis (PCA) from `scikit-learn` is applied to the high-dimensional sentence embeddings. This can help in visualizing embeddings and potentially improving the efficiency of similarity searches, though in this notebook, the similarity search for context retrieval is performed on PCA-transformed embeddings.
3.  **Large Language Models (LLMs) & Hugging Face Pipeline**:
    *   The project utilizes `google/flan-t5-large` from the Hugging Face Hub as the LLM for statement verification.
    *   The `transformers` library's `pipeline` (specifically for `text2text-generation`) provides a high-level API to easily perform inference with this LLM.
    *   `langchain-huggingface`'s `HuggingFacePipeline` is used to integrate this Hugging Face pipeline into a Langchain workflow, simplifying the process of sending prompts (which include the statement and retrieved context) to the LLM and getting back its TRUE/FALSE/NONE determination.
    *   `bitsandbytes` is configured for 8-bit quantization to load the LLM more efficiently, reducing memory footprint while aiming to maintain performance.


## Prerequisites

1.  **Python**: Version 3.9 to 3.12. You can download it from python.org.
    (The notebook `version_2.ipynb` was developed with Python 3.11).
2.  **CUDA**: If you intend to use a GPU (recommended for LLM inference), ensure you have NVIDIA drivers and CUDA Toolkit 11.8 installed. The PyTorch installation is configured for this version.

## Setup Instructions

### 1. Clone the Repository (Optional)
If you haven't already, clone the project repository to your local machine.
```bash
git clone https://github.com/parth1609/Ramayana.git
```

### 2. Create and Activate Virtual Environment
```bash
# Create a virtual environment (e.g., named 'venv')
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 3. Install Python Packages
This project uses `uv` for faster package installation, but `pip` can also be used.

```bash
# Install uv (if not already installed)
pip install uv

# Install core dependencies using uv
uv pip install pandas numpy matplotlib plotly nbformat nltk transformers scikit-learn spacy sentence-transformers accelerate bitsandbytes langchain-huggingface ipython

# Install PyTorch with CUDA 11.8 support
```bash
https://pytorch.org/get-started/locally/
```
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download NLTK Resources
The notebook requires specific NLTK data. You can download them by running the following Python code (or these steps will be attempted automatically when you run the notebook):
```python
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
```



- **Test Suite**: Includes example and batch tests for pipeline evaluation.




## 2. Streamlit App (Frontend) and Modular Backend

### Features
- **Upload Any CSV Dataset**: Upload your own CSV file with Ramayana verses or any text data
- **Flexible Column Selection**: Choose any column from your CSV as the text source
- **Dynamic Processing**: Embeddings and models adapt to your uploaded data
- **Local AI Models**: No API keys required - everything runs locally
- **Single Statement Verification**: Get TRUE/FALSE/NONE answers with retrieved context

### App Structure
The frontend Streamlit app lives in `app.py` with a modular backend under the `ramayana/` package:
- `ramayana/constants.py` — defaults and labels
- `ramayana/data.py` — dataset loading and cleaning
- `ramayana/embeddings.py` — sentence embeddings and PCA
- `ramayana/retrieval.py` — similarity search to fetch contexts
- `ramayana/prompts.py` — prompt templates and rendering
- `ramayana/llm.py` — LLM device/quantization and pipeline loading
- `ramayana/verification.py` — label parsing and verification
- `ramayana/types.py` — typed containers for results

### How to Use

1. **Activate your virtual environment**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the app:**
   ```bash
   streamlit run app.py
   ```
4. **Upload your CSV:** Use the file uploader in the sidebar to upload any CSV file
5. **Select text column:** Choose which column contains your text data (e.g., "English_translation", "Verse", "Text", etc.)
6. **Configure models:** Optionally change the Sentence-Transformer or LLM model
7. **Verify statements:** Enter statements in the main area and get AI-powered verification

### Dataset Requirements
- **Format:** CSV file with at least one text column
- **Content:** Any text data (Ramayana verses, religious texts, stories, etc.)
- **Flexibility:** No fixed column names required - you choose the column
- **Size:** Works with datasets of any size (larger datasets take more time for initial processing)

### Technical Notes
- **Embeddings:** Auto-generated and cached per dataset using content hash
- **GPU Support:** Automatic CUDA detection with 8-bit quantization when available
- **CPU Fallback:** Works entirely on CPU if no GPU available
- **Caching:** Smart caching system - new uploads trigger complete reprocessing
- **No API Keys:** All models run locally (Sentence-Transformers + Hugging Face models)


