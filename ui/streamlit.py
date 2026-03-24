import streamlit as st
from pathlib import Path

from config import BOOKS_DIR


def launch_streamlit_ui(rag_service):
    st.set_page_config(page_title="RAG по книгам", layout="wide")
    st.title("📚 RAG по книгам — Streamlit UI")

    if "query" not in st.session_state:
        st.session_state.query = ""

    query = st.text_input("Введите запрос или вопрос:", st.session_state.query)

    tabs = st.tabs(["Книги", "Поиск", "Вопрос/Ответ", "Добавить книги"])

    # --- Tab 1: Список книг ---
    with tabs[0]:
        st.subheader("📖 Загруженные книги")
        books = rag_service.get_books()
        st.write(f"Всего книг: {len(books)}")
        for b in books:
            st.markdown(f"**{b['title']}** — {len(b['text'])} символов")

    # --- Tab 2: Поиск фрагментов ---
    with tabs[1]:
        st.subheader("🔍 Поиск фрагментов")
        if st.button("Найти фрагменты"):
            if not query.strip():
                st.warning("Введите запрос")
            else:
                result = rag_service.search(query)
                if not result["found"]:
                    st.info(result.get("message", "Ничего не найдено"))
                else:
                    st.success(result["message"])
                    for i, r in enumerate(result["results"], 1):
                        st.markdown("---")
                        st.markdown(f"**#{i} | {r['title']} | chunk {r['chunk_index']}**")
                        st.markdown(f"**Hybrid score:** {r.get('hybrid_score',0):.4f}")
                        st.text(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

    # --- Tab 3: Вопрос-Ответ ---
    with tabs[2]:
        st.subheader("❓ Вопрос-Ответ")
        if st.button("Ответить на вопрос"):
            if not query.strip():
                st.warning("Введите вопрос")
            else:
                result = rag_service.answer(query)
                st.markdown("**ВОПРОС:**")
                st.write(query)
                st.markdown("**ОТВЕТ:**")
                st.write(result.get("answer","Нет ответа"))

                st.markdown("**ЦИТАТЫ:**")
                quotes = result.get("quotes", [])
                if not quotes:
                    st.info("Цитат нет")
                else:
                    for i, q in enumerate(quotes,1):
                        st.markdown("---")
                        st.markdown(f"**#{i} | {q['title']} | chunk {q['chunk_index']} | score {q.get('score',0):.4f}**")
                        st.write(q["text"])

    # --- Tab 4: Добавление книг ---
    with tabs[3]:
        st.subheader("📥 Добавить новые книги (.txt)")
        uploaded_files = st.file_uploader("Добавить книги (.txt)", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                path = BOOKS_DIR / file.name
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

            st.success(f"{len(uploaded_files)} книг загружено!")

            # пересобираем пайплайн
            rag_service.initialize()
            st.info("RAG пайплайн пересобран.")
