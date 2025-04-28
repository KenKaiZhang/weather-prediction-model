import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llms.huggingface_client import HuggingFaceClient

@pytest.fixture
def mock_config():
    return {
        "pretrained_model": "hf-internal-testing/tiny-random-GPT2",
        "model_name": "tiny-gpt2-test"
    }

def test_load_model_success(mock_config):
    client = HuggingFaceClient(config=mock_config)
    client.load()

    # Check that model and tokenizer are loaded
    assert client.model is not None
    assert client.tokenizer is not None
    assert client.model_name == "tiny-gpt2-test"

def test_load_missing_pretrained_model():
    config = {
        "model_name": "tiny-gpt2-test"
    }
    client = HuggingFaceClient(config=config)

    with pytest.raises(ValueError, match="Missing pretrained model from config"):
        client.load()

def test_load_missing_model_name():
    config = {
        "pretrained_model": "hf-internal-testing/tiny-random-GPT2"
    }
    client = HuggingFaceClient(config=config)

    with pytest.raises(ValueError, match="Missing model name from config"):
        client.load()
