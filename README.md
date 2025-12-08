# Applied LLM Engineering Learning Lab

This repository is an educational reconstruction of a six-month learning path in practical LLM engineering.

It covers:

- Python and transformer fundamentals
- Retrieval-Augmented Generation
- LLM evaluation
- QLoRA / PEFT fine-tuning
- Responsible AI assistant prototyping
- Knowledge-graph memory for agents
- Deployment basics with Streamlit, Docker, and Kubernetes notes

> Transparency note: this repository is organized retrospectively as a learning archive. Some commits may use historical dates to structure the educational timeline. It should not be interpreted as a real-time work log.

## Repository Structure

```text
01_foundations/
02_transformers_basics/
03_rag_from_scratch/
04_llm_evaluation/
05_qlora_finetuning/
06_responsible_ai_assistant/
07_knowledge_graph_memory/
08_deployment_basics/
assets/
```

## How to use

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run selected demos:

```bash
streamlit run 03_rag_from_scratch/app.py
streamlit run 04_llm_evaluation/streamlit_app.py
streamlit run 06_responsible_ai_assistant/demo_app.py
streamlit run 07_knowledge_graph_memory/app.py
```

## Learning Timeline

See [`timeline.md`](timeline.md).
