from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "model": (["civitai-nsfw-500k-3e4", "civitai-sfw-500k-3e4"], {}),
                "max_tokens": (
                    "INT",
                    {"default": 100, "min": 1, "max": 500, "step": 10},
                ),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 500, "step": 10}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "make_text"
    OUTPUT_NODE = True
    CATEGORY = "llm"

    def load_model(self, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        # model.to("cuda")
        return model

    def load_tokenizer(self, tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def generate_text(self, model, sequence, max_length, top_k=50, top_p=0.95):
        output_dir = f"/home/ssube/notebooks/gpt2/training-{model}"
        model = self.load_model(output_dir)
        tokenizer = self.load_tokenizer(output_dir)
        ids = tokenizer.encode(sequence, return_tensors="pt")
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=top_k,
            top_p=top_p,
        )
        result = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        return result

    def make_text(self, prompt, model, max_tokens, top_k, top_p):
        output = self.generate_text(model, prompt, max_tokens, top_k, top_p)
        return {
            "result": [output],
            "ui": {"text": [output]},
        }
