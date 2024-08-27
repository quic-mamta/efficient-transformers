import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from QEfficient.transformers.transform import transform_lm
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner


def load_pytorch_model(model_name, model_class):
    model_path = hf_download(
        repo_id=model_name, ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"]
    )
    model_hf = model_class.from_pretrained(model_path, num_hidden_layers=1, use_cache=True)
    model_hf.eval()
    return model_hf


def transform_pt_model_with_qeff(model_hf):
    model_kv = transform_lm(model_hf)
    model_kv.eval()
    return model_kv


model_name = "meta-llama/Llama-2-7b-chat-hf"
prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
prompt = "USER: " + prompt + "\n\nASSISTANT: "
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


model_hf = load_pytorch_model(model_name, LlamaForCausalLM)
# hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

model_kv = transform_pt_model_with_qeff(model_hf)

api_runner = ApiRunner(
    1,
    tokenizer,
    model_kv.config,
    prompt,
    Constants.PROMPT_LEN,
    65,  # ctx_len
)

kv_tokens = api_runner.run_kv_model_on_pytorch(model_kv)
"""
onnx_path = "qeff_models/fp16.onnx"
ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

qpc_path = "qeff_models/qpcs"
from QEfficient.generation.cloud_infer import QAICInferenceSession
session = QAICInferenceSession(qpc_path, [4], enable_debug_logs=False)
cloud_ai_100_tokens = api_runner.run_kv_model_on_cloud_ai_100(session, 32, [1,32,65, 128])
"""
