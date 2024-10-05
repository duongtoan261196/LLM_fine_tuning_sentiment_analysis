import streamlit as st
import time
import re
from PIL import Image
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################## Page settup ##########################################################
st.set_page_config(layout="wide")

st.sidebar.title("Control Panel")

with st.sidebar:
    image = Image.open('./Figs/streamlit_figs/chatbot_logo.jpeg')
    st.image(image, width=200)

st.sidebar.markdown("### Contact Information")
st.sidebar.write("For inquiries, please contact:")
st.sidebar.write("Le Toan DUONG")
st.sidebar.write("le.toan.duong.11@gmail.com")


st.markdown("<h1 style ='text-align: center;'> Welcome to my Chatbot </h1>", unsafe_allow_html=True)

#####################################################################################################################
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
greeting_string = ['hi', 'hello', 'good morning', 'good afternoon', 'good evening', 'hey']

def reset_conversation():
  st.session_state.messages = []



@st.cache_data
def load_data():    
    if st.session_state["openai_model"] == 'llama3.2-1B':
        model_name = "meta-llama/Llama-3.2-1B"
        finetuned_model = "trained_weights/llama3-2"
    else:
        model_name = "openai-community/gpt2-large"
        finetuned_model = "trained_weights/gpt2-large"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = "right"

    model = AutoPeftModelForCausalLM.from_pretrained(
        finetuned_model,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    merged_model = model.merge_and_unload()

    return tokenizer, merged_model


# Streamed response emulator
def response_generator(prompt):
    if any(substring in prompt.lower().split(' ') for substring in greeting_string):
        response = "Hello. Please input a news with the following template to get the sentiment analysis result. Title: [], Content: []"
    else:
        # tokenizer, merged_model = load_data()
        try:
            title, content = re.findall(r'\[(.*?)\]', prompt)
            llm_prompt = f""" 
            # Title: {title}
            # Text: {content}
            # Constraints: Generate the label from "positive" or "negative" or "neutral" with the delimiter "Prediction".
            # Prediction= 
            # """.strip()
        except:
            response = 'Oups! Please input with the template: Title: [], Content: []'
        else:
            # pipe = pipeline(task="text-generation", 
            #         model=merged_model, 
            #         tokenizer=tokenizer, 
            #         max_new_tokens = 1, 
            #         temperature = 0.001,
            #         )
            # result = pipe(llm_prompt)
            # response = result[0]['generated_text'].split("=")[-1].lower()
        
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"{llm_prompt}"
                    }
                ]
            )
      
            answer = completion.choices[0].message.content
            response = predict(answer)
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def predict(answer):
    answer = answer.lower()
    if "positive" in answer:
        y_pred = "positive"
    elif "negative" in answer:
        y_pred = "negative"
    elif "neutral" in answer:
        y_pred = "neutral"
    else:
        y_pred = "neutral"
    return y_pred

mode = st.sidebar.selectbox(
    "How would you like use the chatbot :robot_face:?",
    ("Conversation", "Sentiment analysis"),
)
if mode == "Conversation":
    list_models = ['gpt-3.5-turbo-0125', 'gpt-4-turbo', 'gpt-4', 'o1-preview', 'o1-mini', 'gpt-4o-mini', 'gpt-4o']
else:
    list_models = ['llama3.2-1B', 'gpt2-large']
model = st.sidebar.selectbox("Select a LLM model", list_models)
st.session_state["openai_model"] = model

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
st.sidebar.button('Reset Chat', on_click=reset_conversation)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt:= st.chat_input("Chat here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if mode == "Conversation":
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                stream=True
            )
            response = st.write_stream(stream)
        elif mode == 'Sentiment analysis':
            response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})