import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
from config import BOOKS_DIR, TOP_K_SEARCH, TOP_K_QA

class RAGDesktopApp:
    def __init__(self, root, rag_service):
        self.root = root
        self.rag = rag_service
        self.root.title("RAG по книгам")
        self.root.geometry("1200x800")
        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Запрос или вопрос:").grid(row=0, column=0, sticky="w")
        self.entry = ttk.Entry(top)
        self.entry.grid(row=1, column=0, sticky="ew", pady=6)

        btns = ttk.Frame(top)
        btns.grid(row=2, column=0, sticky="w")
        ttk.Button(btns, text="Поиск", command=self.search).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Ответ", command=self.answer).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Книги", command=self.show_books).grid(row=0, column=2, padx=5)
        ttk.Button(btns, text="Добавить книги", command=self.add_books).grid(row=0, column=3, padx=5)

        output_frame = ttk.Frame(self.root, padding=10)
        output_frame.grid(row=1, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD)
        self.output.grid(row=0, column=0, sticky="nsew")

    # --- actions ---
    def _write(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def _clear(self):
        self.output.delete("1.0", tk.END)

    def show_books(self):
        self._clear()
        books = self.rag.get_books()
        self._write(f"Книг: {len(books)}\n")
        for b in books:
            self._write(f"- {b['title']} ({len(b['text'])} символов)")

    def search(self):
        query = self.entry.get().strip()
        if not query:
            messagebox.showwarning("Ошибка", "Введите запрос")
            return
        self._clear()
        try:
            result = self.rag.search(query, TOP_K_SEARCH)
            self._write(result.get("message", ""))
            for i, r in enumerate(result.get("results", []), 1):
                self._write("="*80)
                self._write(f"#{i} | {r['title']} | chunk {r['chunk_index']}")
                self._write(f"score: {r.get('hybrid_score',0):.4f}")
                self._write(r["text"][:800])
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def answer(self):
        question = self.entry.get().strip()
        if not question:
            messagebox.showwarning("Ошибка", "Введите вопрос")
            return
        self._clear()
        try:
            result = self.rag.answer(question, TOP_K_QA)
            self._write("ВОПРОС:\n" + question + "\n")
            self._write("ОТВЕТ:\n" + result.get("answer","") + "\n")
            self._write("\nЦИТАТЫ:\n")
            for i, q in enumerate(result.get("quotes", []), 1):
                self._write("-"*80)
                self._write(f"{i}. {q['title']} (chunk {q['chunk_index']})")
                self._write(q["text"])
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def add_books(self):
        files = filedialog.askopenfilenames(title="Выберите книги (.txt)", filetypes=[("Text files","*.txt")])
        if not files:
            return
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        for f in files:
            path = BOOKS_DIR / Path(f).name
            with open(f, "rb") as src, open(path, "wb") as dst:
                dst.write(src.read())
        messagebox.showinfo("Готово", f"{len(files)} книг добавлено!")
        self.rag.initialize()  # пересобираем RAG
        self.show_books()

def launch_desktop_app(rag_service):
    root = tk.Tk()
    app = RAGDesktopApp(root, rag_service)
    root.mainloop()