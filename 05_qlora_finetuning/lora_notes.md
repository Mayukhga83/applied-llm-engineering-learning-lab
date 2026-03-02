# LoRA and QLoRA Notes

LoRA trains small low-rank adapter matrices instead of updating all model weights.

QLoRA combines:

- 4-bit quantization of the base model
- LoRA adapters
- lower memory training
- instruction tuning on consumer or rented GPUs

Typical workflow:

1. Choose base model
2. Prepare instruction dataset
3. Format examples with chat template
4. Load model in 4-bit
5. Attach LoRA adapters
6. Train
7. Save adapter
8. Upload to Hugging Face
9. Test with inference script
