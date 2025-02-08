import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

st.title("AI Studio")

system_instruction = st.text_area("Enter System Instruction:", "You are a helpful assistant. Please respond to user queries.")

llm_model = st.sidebar.selectbox("Select Open Source model", ["mistral", "gemma", "deepseek-r1"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

stop_sequence = st.sidebar.text_input("Stop Sequence:", value="END")
output_length = st.sidebar.number_input("Output Length:", min_value=1, max_value=500, value=150)

st.sidebar.header("Safety Settings")
enable_filtering = st.sidebar.checkbox("Enable Content Filtering", value=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

def generate_response(question, chat_history, instruction):
    """Generate response while considering chat history."""
    llm = Ollama(model=llm_model)
    output_parser = StrOutputParser()

    full_prompt = ChatPromptTemplate.from_messages(
        [("system", instruction)]
        + [("user", msg["content"]) if msg["role"] == "user" else ("assistant", msg["content"]) for msg in chat_history]
        + [("user", question)]
    )

    chain = full_prompt | llm | output_parser
    answer = chain.invoke({
        "stop_sequence": stop_sequence,
        "max_tokens": output_length,
        "enable_filtering": enable_filtering,
    })
    return answer.strip()

if user_input := st.chat_input("Ask a question..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    response = generate_response(user_input, st.session_state["messages"], system_instruction)
    st.session_state["messages"].append({"role": "assistant", "content": response})

    st.chat_message("assistant").write(response)