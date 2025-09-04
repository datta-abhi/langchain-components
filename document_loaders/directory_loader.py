from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(path = 'document_loaders/books', glob= '*.pdf', loader_cls=PyMuPDFLoader)

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)