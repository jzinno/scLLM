import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import math
import torch

# model_name = "generative/models/scT5proto-eso-52k"
model_name = "generative/models/Tabula5apiens-1epoch"
max_input_length = 512
model_basename = model_name.split("/")[-1]

st.header("Generate candidate cell types for cell sentences")

st_model_load = st.text("Loading cell type generator model...")


@st.cache_resource
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nltk.download("punkt")
    print(f"Model: {model_basename} loaded!")
    return tokenizer, model


tokenizer, model = load_model()
st.success(f"Model: {model_basename} loaded!", icon="🦾")
st_model_load.text("")

with st.sidebar:
    st.header("Model parameters")
    if "num_predictions" not in st.session_state:
        st.session_state.num_predictions = 5

    def on_change_num_predictions():
        st.session_state.num_predictions = num_predictions

    num_predictions = st.slider(
        "Number of cell types to generate",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        on_change=on_change_num_predictions,
    )
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    def on_change_temperatures():
        st.session_state.temperature = temperature

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.6,
        step=0.05,
        on_change=on_change_temperatures,
    )
    st.markdown("_High temperature means that results are more random_")
    if "top_k" not in st.session_state:
        st.session_state.top_k = 10

    def on_change_top_k():
        st.session_state.top_k = top_k

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        on_change=on_change_top_k,
    )
    st.markdown("_Top K means that only the top K most probable tokens are considered_")

if "task" not in st.session_state:
    st.session_state.task = "label cell type: "
st_task = st.text_input(
    "Model task",
    value=st.session_state.task,
    placeholder="label cell type: ",
)


if "text" not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area(
    "Text to generate the cell type for", value=st.session_state.text, height=500
)


def generate_cell_type():
    st.session_state.text = st_text_area

    st.session_state.task = st_task

    # tokenize text
    inputs = [st_task + st_text_area]
    inputs = tokenizer(inputs, return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    print(f"Input has {num_tokens} tokens")
    max_input_length = 500
    num_spans = math.ceil(num_tokens / max_input_length)
    print(f"Input has {num_spans} spans")
    overlap = math.ceil(
        (num_spans * max_input_length - num_tokens) / max(num_spans - 1, 1)
    )
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append(
            [start + max_input_length * i, start + max_input_length * (i + 1)]
        )
        start -= overlap
    print(f"Span boundaries are {spans_boundaries}")
    spans_boundaries_selected = []
    j = 0
    for _ in range(num_predictions):
        spans_boundaries_selected.append(spans_boundaries[j])
        j += 1
        if j == len(spans_boundaries):
            j = 0
    print(f"Selected span boundaries are {spans_boundaries_selected}")

    # transform input with spans
    tensor_ids = [
        inputs["input_ids"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries_selected
    ]
    tensor_masks = [
        inputs["attention_mask"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries_selected
    ]

    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks),
    }

    # compute predictions
    outputs = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_k=top_k, num_beams=8
    )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predicted_cell_types = [
        nltk.sent_tokenize(decoded_output.strip())[0]
        for decoded_output in decoded_outputs
    ]

    st.session_state.cell_types = predicted_cell_types


# generate cell_type button
st_generate_button = st.button("Generate Cell Type", on_click=generate_cell_type)

# cell_type generation labels
if "cell_types" not in st.session_state:
    st.session_state.cell_types = []

if len(st.session_state.cell_types) > 0:
    with st.container():
        st.subheader("Generated Cell Types")
        for cell_type in st.session_state.cell_types:
            st.markdown("__" + cell_type + "__")
