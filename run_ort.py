import numpy as np
import onnx
import onnxruntime
import transformers

model_name = "meta-llama/Llama-2-7b-chat-hf"
n_layer = 32
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
prompt = "5 things to do in India?"
#prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
prompt = "USER: " + prompt + "\n\nASSISTANT: "
print("\n" + prompt, end="")


def prepare_ort_inputs(n_layer=n_layer, ctx_len=65):
    inputs = tokenizer(
        prompt,
        return_tensors="np",
    )
    batch_size, input_len = inputs["input_ids"].shape
    inputs.pop("attention_mask")
    inputs["position_ids"] = np.arange(input_len).reshape(batch_size, -1)

    for i in range(n_layer):
        inputs[f"past_key.{i}"] = np.zeros((1, 32, ctx_len, 128), dtype=np.float32)
        inputs[f"past_value.{i}"] = np.zeros((1, 32, ctx_len, 128), dtype=np.float32)
        inputs[f"past_scores.{i}"] = np.zeros((1, 32, ctx_len), dtype=np.float32)
    return inputs


def update_ort_inputs(inputs, ort_outputs, n_layer=n_layer, window_length=64):
    updated_inputs = {}
    updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
    if inputs["position_ids"][0][0] < 64:
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
    else:
        updated_inputs["position_ids"] = inputs["position_ids"]
    for i in range(n_layer):
        updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][3 * i]
        updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][3 * i + 1]
        updated_inputs["past_scores." + str(i)] = ort_outputs["past_key_values"][3 * i + 2]
    return updated_inputs


def run_ort_session(inputs, session, n_layer=n_layer):
    outputs = {}
    output_names = [x.name for x in session.get_outputs()]
    session_input_names = [x.name for x in session.get_inputs()]
    session_inputs = {}
    for inp_name in session_input_names:
        if inp_name in inputs.keys():
            session_inputs[inp_name] = inputs[inp_name]
    outputs_data = session.run(output_names, session_inputs)
    ort_outputs = dict(zip(output_names, outputs_data))

    present_key_values = []
    for i in range(n_layer):
        if "past_key." + str(i) + "_RetainedState" in ort_outputs:
            present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
        if "past_value." + str(i) + "_RetainedState" in ort_outputs:
            present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])
        if "past_scores." + str(i) + "_RetainedState" in ort_outputs:
            present_key_values.append(ort_outputs["past_scores." + str(i) + "_RetainedState"])

    outputs["past_key_values"] = present_key_values
    outputs["logits"] = ort_outputs["logits"]

    return outputs


def run_kv_model_on_ort(model_path, n_layer=n_layer, ctx_len=65):
    m = onnx.load(model_path, load_external_data=False)
    for node in m.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t)
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                node.attribute[0].t.raw_data = np.array(-1).tobytes()

    onnxruntime_model = model_path[:-5] + "_ort.onnx"
    onnx.save(m, onnxruntime_model)
    session = onnxruntime.InferenceSession(onnxruntime_model)

    generated_ids = []
    inputs = prepare_ort_inputs(n_layer, ctx_len=65)
    # print(0, inputs["input_ids"], inputs["position_ids"])
    ort_outputs = run_ort_session(inputs, session, n_layer)

    print(ort_outputs.keys())

    for i in range(1, 100):
        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        # print(generated_ids[-1])
        inputs = update_ort_inputs(inputs, ort_outputs, n_layer)
        # print(i, inputs["input_ids"], inputs["position_ids"])
        ort_outputs = run_ort_session(inputs, session, n_layer)

    generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    # print(generated_ids[-1])
    generated_ids = np.concatenate(generated_ids, axis=1)
    predicted_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Completion:", repr(predicted_string))
    return generated_ids


model_path = "/home/mamtsing/.cache/qeff_models/LlamaForCausalLM-4400e2ebeb6e0c9f/LlamaForCausalLM.onnx"#"/home/ubuntu/.cache/qeff_models/LlamaForCausalLM-6c541cf409be0bec/LlamaForCausalLM.onnx"
run_kv_model_on_ort(model_path, n_layer=n_layer, ctx_len=65)
