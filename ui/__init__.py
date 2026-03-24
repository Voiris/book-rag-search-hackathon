from .desktop import launch_desktop_app

def load_streamlit():
    from .streamlit import launch_streamlit_ui
    return launch_streamlit_ui

__all__ = [
    "launch_desktop_app",
    "load_streamlit"
]
