import os
from typing import List
from langchain_core.documents import Document


def load_markdown_folder(path: str) -> List[Document]:
    docs = []

    print(f"📂 Loading from: {path}")

    if not os.path.exists(path):
        print("Path does not exist!")
        return docs

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)

                print(f" Found: {full_path}")

                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": full_path}
                    )
                )

    return docs


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "rag_db")
    )

    print("CWD:", os.getcwd())
    print("BASE_DIR:", BASE_DIR)
    print("DATA_PATH:", data_path)

    documents = load_markdown_folder(data_path)

    print(f"\nTotal documents loaded: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc.metadata['source']}")


if __name__ == "__main__":
    main()
