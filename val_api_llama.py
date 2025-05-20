import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from QEfficient.transformers.transform import transform_lm
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

'''
model_name = "meta-llama/Llama-2-7b-chat-hf"
prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
prompt = "USER: " + prompt + "\n\nASSISTANT: "
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


model_path = hf_download(repo_id=model_name, ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"])
model_hf = LlamaForCausalLM.from_pretrained(model_path, num_hidden_layers=1, use_cache=True)
model_hf.eval()

api_runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=model_hf.config,
    prompt=prompt,
    prompt_len=Constants.PROMPT_LEN,
    ctx_len=65,
)

#hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

model_kv = transform_lm(model_hf)

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
qeff_model = AutoModelForCausalLM.from_pretrained(model_name)


kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
'''

from transformers import AutoModelForCausalLM
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.run_utils import ApiRunner

def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params
    
model_name = "meta-llama/Llama-2-7b-chat-hf"
model_config = {"model_name": model_name}
model_config["n_layer"] = 1
model_hf, _ = load_causal_lm_model(model_config)

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
config = model_hf.config
prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
prompt = "USER: " + prompt + "\n\nASSISTANT: "
api_runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=prompt,
    prompt_len=Constants.PROMPT_LEN,
    ctx_len=65,
)

#pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
qeff_model = QEFFAutoModelForCausalLM(model_hf, is_tlm=False, pretrained_model_name_or_path=model_name)
pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

#assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
#    "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
#) 


"""
onnx_path = "qeff_models/fp16.onnx"
ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

qpc_path = "qeff_models/qpcs"
from QEfficient.generation.cloud_infer import QAICInferenceSession
session = QAICInferenceSession(qpc_path, [4], enable_debug_logs=False)
cloud_ai_100_tokens = api_runner.run_kv_model_on_cloud_ai_100(session, 32, [1,32,65,128])
"""
