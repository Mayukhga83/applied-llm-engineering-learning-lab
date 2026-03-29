# Modern Post-Training

This module extends the QLoRA introduction into an evaluation-first post-training workflow:

1. establish an untuned baseline;
2. run supervised fine-tuning (SFT);
3. optimize labeled preferences with DPO;
4. define deterministic, verifiable rewards;
5. run a small GRPO experiment;
6. compare every checkpoint on held-out cases.

The checked-in scripts validate data and configurations without downloading a model. Actual training requires the optional Hugging Face `transformers`, `datasets`, `peft`, and `trl` packages plus suitable compute.
