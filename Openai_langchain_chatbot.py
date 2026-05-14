import os
import anthropic
import streamlit as st

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_claude_response(messages: list) -> str:
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=16000,
        system="You are a helpful assistant providing clear and accurate answers.",
        messages=messages,
        cache_control={"type": "ephemeral"},
    ) as stream:
        return stream.get_final_message().content[0].text


def main():
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("PGX Chatbot")

    with col2:
        image_path = "C:/Users/danyh/Desktop/coop/شعار-مدينة-الملك-عبدالعزيز-للعلوم-والتقنية.png"
        if os.path.exists(image_path):
            st.image(image_path, width=150)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi! I'm here to answer your questions about PGX"}
        ]

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("Ask a question about PGX:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_claude_response(api_messages)
            st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
