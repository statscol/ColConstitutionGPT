import gradio as gr
from common import DATA
from config import DEFAULT_BOT_MESSAGE
from inference_hf import rag_chatbot


DATA = DATA.add_faiss_index("embedding")
DEFAULT_MESSAGE = "Haz aquí tu pregunta"


async def predict(message, chat_history):
    bot_message = rag_chatbot(message, k=3)
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks(theme=gr.themes.Base()) as demo:
    chatbot = gr.Chatbot(
        value=[[None, DEFAULT_BOT_MESSAGE]], label="ReformaPensional-Llama3"
    )
    msg = gr.Textbox(placeholder=DEFAULT_MESSAGE)
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(predict, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_port=9090)
