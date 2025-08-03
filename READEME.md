# 🚀 LangChain Project: Multimodal LLM Playground

This repository provides a modular, extensible playground for experimenting with multiple **LLMs** using [LangChain](https://docs.langchain.com/). It covers **OpenAI**, **Anthropic**, **Google Gemini**, **Hugging Face (cloud & local)**, and more.

Whether I'm testing **LLMs**, **embeddings**, **prompts**, **chains**, or building full NLP pipelines, this repo has hands-on examples and utilities ready to go!

---

## 📁 Project Structure

```
├── 1.langchain-models
│   ├── 1.LLMs
│   │   └── 1_llm_demo.py
│   ├── 2.ChatModels
│   │   ├── 1_chatmodel_opemai.py
│   │   ├── 1_chatmodel_openai.py
│   │   ├── 2_chatmodel_anthropic.py
│   │   ├── 3_chatmodel_google.py
│   │   ├── 4_chatmodel_hf_api.py
│   │   └── 5_chatmodel_hf_local.py
│   ├── 3.EmbeddingModels
│   │   ├── 1_embedding_openai_query.py
│   │   ├── 2_embedding_openai_docs.py
│   │   ├── 3_embedding_hf_local.py
│   │   ├── 4_document_similarity.py
│   │   └── 5_document_similarity_local.py
│   └── 4.LainChainPrompts
│       ├── 1.2_prompt_ui_hf_template.py
│       ├── 1_local_prompt_ui_hf.py
│       ├── 1_prompt_ui_hf.py
│       ├── 1_prompt_ui_openAi.py
│       ├── 2_prompt_ui_openAi.py
│       └── prompt_generator.py
├── 2.langchain-prompts
│   ├── chatbot.py
│   ├── chat_history.txt
│   ├── chat_prompt_template.py
│   ├── message_placeholder.py
│   ├── messages.py
│   ├── prompt_generator.py
│   ├── prompt_template.py
│   ├── prompt_ui.py
│   ├── temperature.py
│   └── template.json
├── 3.langchain-structured-output
│   ├── json_schema.json
│   ├── pydantic_demo.py
│   ├── students_dataset.csv
│   ├── typeddict_demo.py
│   ├── with_structured_output_json.py
│   ├── with_structured_output_llama.py
│   ├── with_structured_output_pydantic.py
│   └── with_structured_output_typeddict.py
├── 4.langchain-output-parsers
│   ├── jsonoutputparser.py
│   ├── pydanticoutputparser.py
│   ├── stroutputparser1.py
│   ├── stroutputparser.py
│   └── structuredoutputparser.py
├── 5.langchain-chains
│   ├── conditional_chain.py
│   ├── parallel_chain.py
│   ├── sequential_chain.py
│   └── simple_chain.py
├── 6.langchain-runnables
│   ├── runnable_branch.py
│   ├── runnable_lambda.py
│   ├── runnable_parallel.py
│   ├── runnable_passthrough.py
│   └── runnable_sequence.py
├── 7.langchain-document-loaders
│   ├── books
│   │   ├── Building Machine Learning Systems with Python - Second Edition.pdf
│   │   └── readme.md
│   ├── cricket.txt
│   ├── csv_loader.py
│   ├── directory_loader.py
│   ├── dl-curriculum.pdf
│   ├── pdf_loader.py
│   ├── Social_Network_Ads.csv
│   ├── text_loader.py
│   └── webbase_loader.py
├── 8.langchain-text-splitters
│   ├── dl-curriculum.pdf
│   ├── length_based.py
│   ├── markdown_splitting.py
│   ├── python_code_splitting.py
│   ├── semantic_meaning_based.py
│   └── text_structure_based.py
├── GenAI_Presntations.pdf
├── LangChain_Notes.pdf
├── LangChain_Project
│   ├── 1.LLMs
│   │   └── 1_llm_demo.py
│   ├── 2.ChatModels
│   │   ├── 1_chatmodel_opemai.py
│   │   ├── 2_chatmodel_anthropic.py
│   │   ├── 3_chatmodel_google.py
│   │   ├── 4_chatmodel_hf_api.py
│   │   └── 5_chatmodel_hf_local.py
│   ├── 3.EmbeddingModels
│   │   ├── 1_embedding_openai_query.py
│   │   ├── 2_embedding_openai_docs.py
│   │   ├── 3_embedding_hf_local.py
│   │   ├── 4_document_similarity.py
│   │   └── 5_document_similarity_local.py
│   ├── 4.LainChainPrompts
│   │   ├── 1.2_prompt_ui_hf_template.py
│   │   ├── 1_local_prompt_ui_hf.py
│   │   ├── 1_prompt_ui_hf.py
│   │   ├── 1_prompt_ui_openAi.py
│   │   ├── 2_prompt_ui_openAi.py
│   │   └── prompt_generator.py
│   ├── requirement.txt
│   └── template.json
├── requirement.txt
└── template.json               # JSON prompt templates

````

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/langchain-project.git
cd langchain-project
````

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

> Note: All modules are compartmentalized. You can run individual scripts without needing the entire repo loaded in memory.

---

## 🔐 Environment Configuration (`.env`)

Create a `.env` file in the project root with your API keys and configs. Below is a sample configuration:

```env
# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Anthropic (Claude)
ANTHROPIC_API_KEY=your-anthropic-key

# Google Gemini
GOOGLE_API_KEY=your-google-api-key

# HuggingFace API (cloud)
HUGGINGFACEHUB_API_TOKEN=your-huggingface-api-key

# Local HuggingFace Model (optional - if using local model)
HF_MODEL_NAME=gpt2
HF_MODEL_PATH=models/gpt2  # Local path or model name

# Optional Configuration
TEMPERATURE=0.7
MAX_TOKENS=512
```

> If you're using **local HuggingFace models**, ensure the model is cached locally or downloaded to the specified `HF_MODEL_PATH`.

---

## ✅ Usage Examples

Each subfolder contains purpose-driven examples. Here's how to run a few popular ones:

### 🔸 OpenAI Chat Model

```bash
python 1.langchain-models/2.ChatModels/1_chatmodel_openai.py
```

### 🔸 HuggingFace Local Model Chat

```bash
python 1.langchain-models/2.ChatModels/5_chatmodel_hf_local.py
```

### 🔸 Embeddings + Similarity Search

```bash
python 1.langchain-models/3.EmbeddingModels/4_document_similarity.py
```

### 🔸 Structured Output using Pydantic

```bash
python 3.langchain-structured-output/with_structured_output_pydantic.py
```

### 🔸 Document Loading (e.g., PDFs, CSVs)

```bash
python 7.langchain-document-loaders/pdf_loader.py
```

---

## 🧠 Local Model Setup (Optional)

To use a **local HuggingFace model**, no API key is needed. Just set:

```env
HF_MODEL_NAME=gpt2
HF_MODEL_PATH=models/gpt2
```

Ensure it's downloaded using:

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Or manually download using:

```bash
huggingface-cli download gpt2
```

---

## 📎 Extra Resources

* `LangChain_Notes.pdf` → High-level summary of concepts
* `GenAI_Presntations.pdf` → Slide deck for presentations
* `template.json` → Prompt templates used in various examples

---

## 🤝 Contribution

Pull requests are welcome! Feel free to fork, explore new LLM APIs, or add more document loaders and prompt strategies.

---

## 📜 License

This repository is open-sourced under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

Built using:

* [LangChain](https://github.com/langchain-ai/langchain)
* [Hugging Face](https://huggingface.co/)
* [Anthropic Claude](https://www.anthropic.com/)
* [Google Gemini](https://ai.google.dev/)
* [OpenAI](https://platform.openai.com/)
* [DSMP 2.O](https://docs.google.com/document/d/1OsMe9jGHoZS67FH8TdIzcUaDWuu5RAbCbBKk2cNq6Dk/edit?tab=t.0)

---
## ✨ Colab Code
* [Langchain Aam Zindagi](https://colab.research.google.com/drive/1gv3e-OfHCi6IuVBVR7xWmrcVl6gHsydv?usp=sharing)
* [langchain Mentos Zindagi](https://colab.research.google.com/drive/1P11hpAPtjr0oIiqQ5pNsdnMxmascdbBY?usp=sharing)
* [Langchain Chroma Vector Database](https://colab.research.google.com/drive/1VwOywJ9LPSIpKWKj9vueVoexSCzGHXNC?usp=sharing)
* [Langchain Retrievers](https://colab.research.google.com/drive/1vuuIYmJeiRgFHsH-ibH_NUFjtdc5D9P6?usp=sharing)

---
```
---
Vilal Ali
vilal.ali@research.iiit.ac.in
MS By Research, IIITH
```
