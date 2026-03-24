import os

from config import *
from services.rag_service import RAGService


def is_running_under_streamlit() -> bool:
    return "STREAMLIT_RUN_CONTEXT" in os.environ or "STREAMLIT_SERVER_PORT" in os.environ

def main():
    rag = RAGService(
        books_dir=BOOKS_DIR,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

    rag.initialize()

    if is_running_under_streamlit():
        from ui import load_streamlit
        launch_streamlit_ui = load_streamlit()
        launch_streamlit_ui(rag)
    else:
        from ui.desktop import launch_desktop_app
        launch_desktop_app(rag)

if __name__ == "__main__":
    main()
