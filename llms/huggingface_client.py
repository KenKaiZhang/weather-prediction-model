import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
)

from llms.base_client import BaseClient


class HuggingFaceClient(BaseClient):

    def load(self):

        model_name = self.model_config.get("name")
        use_lora = self.model_config.get("use_lora", False)
        pretrained_model = self.model_config.get("pretrained")

        if model_name is None:
            raise ValueError("Missing model name from config")
        if pretrained_model is None:
            raise ValueError("Missing pretrained model from config")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        if use_lora:
            lora_config = LoraConfig(
                r=int(self.lora_config.get("r", 8)),
                lora_alpha=int(self.lora_config.get("alpha", 32)),
                lora_dropout=float(self.lora_config.get("dropout", 0.1)),
                target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
            print(f"Successfully loaded {model_name} with LoRA enabled")
        else:
            self.model = base_model
            print(f"Successfully loaded {model_name} without LoRA")

    def train(self, train_dataset, val_dataset=None):

        checkpoint_dir = self.training_config.get("checkpoint_dir")
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=500,
            logging_steps=100,
            save_steps=500,
            per_device_train_batch_size=int(self.training_config.get("batch_size", 1)),
            gradient_accumulation_steps=int(self.training_config.get("grad_acc_steps", 4)),
            num_train_epochs=int(self.training_config.get("epochs", 3)),
            learning_rate=float(self.training_config.get("learning_rate", 2e-5)),
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=f"{checkpoint_dir}/logs",
            save_total_limit=2,
            fp16=True,  # if you have GPU
            push_to_hub=False,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            label_names=["labels"],
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        trainer.train()
        trainer.save_model(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
