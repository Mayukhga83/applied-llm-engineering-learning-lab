# Applied LLM Engineering Learning Lab

This repository is an educational reconstruction of a learning path in practical LLM engineering.

It covers:

- Python and transformer fundamentals
- Retrieval-Augmented Generation
- LLM evaluation
- QLoRA / PEFT fine-tuning
- Responsible AI assistant prototyping
- Knowledge-graph memory for agents
- Deployment basics with Streamlit, Docker, and Kubernetes notes
- Structured generation and tool contracts
- Hybrid retrieval, reranking, and retrieval evaluation
- Bounded agents and Model Context Protocol integration
- Evaluation-driven development, security, and observability
- Modern post-training with preference and verifiable-reward methods

> Transparency note: this repository is organized retrospectively as a learning archive. Some commits use historical author and committer dates to structure the educational timeline, including the advanced track reconstructed across February and March 2026. It should not be interpreted as a real-time work log.

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
09_structured_generation/
10_advanced_rag/
11_agent_engineering/
12_eval_driven_development/
13_agent_security_observability/
14_modern_post_training/
common/
evals/
benchmarks/
reports/
tests/
assets/
```

## Advanced track

The advanced track treats evaluation as infrastructure rather than a final check. Each module contains a small, provider-neutral implementation, executable examples, tests, and measurable acceptance criteria. Optional integrations are isolated behind interfaces so the core exercises run locally.

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
