import os
import requests

class BaseClient:

    def __init__(self, config):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.lora_config = config.get("lora", {})
        
        self.model = None
        self.model_name = None
        self.tokenizer = None

    def load(self) -> None:
        raise NotImplementedError("load_model() must be implemented in the subclass")
    
    def train(self) -> None:
        raise NotImplementedError("train_model() must be implemented in the subclass")
    
    def save(self) -> None:
        models_dir = os.getenv("MODELS_DIR")
        model_path = os.path.join(models_dir, self.model_name)

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        url = os.getenv("OLLAMA_URL")
        payload = {
            "path": model_path,
            "name": self.model_name
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Successfully registerd {self.model_name} with Ollama!")
        else:
            raise Exception(f"Failed to register {self.model_name} with Ollama: {response.text}")