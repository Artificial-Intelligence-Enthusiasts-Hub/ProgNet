from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models.gigachat import GigaChat

# Define the AI class
class ModularAI:

    # Initialization of the class
    def __init__(self, credentials):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Load documents from directory
        self.docs = self._load_documents_from_directory("docs")
        
        # Create documents after text splitting
        self.documents = self._split_text(self.docs)
        
        # Embed the documents
        self.embedder = SpacyEmbeddings(model_name="en_core_web_sm")
        
        # Initialize the Chroma database
        self.db = self._initialize_chroma(self.documents)
        
        # Instantiate the chat model
        self.chat_model = GigaChat(profanity_check=False, verify_ssl_certs=False, credentials=credentials)

        # Create a chain for retrieval-based QA
        self.qa_chain = RetrievalQA.from_chain_type(self.chat_model, retriever=self.db.as_retriever())

    # Method to load documents from directory
    def _load_documents_from_directory(self, directory_path):
        directory_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader,
                                           show_progress=True, use_multithreading=True, silent_errors=True)
        return directory_loader.load()

    # Method to split text
    def _split_text(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)

    # Method to initialize chroma vector store
    def _initialize_chroma(self, documents):
        return Chroma.from_documents(documents, self.embedder, client_settings=Settings(anonymized_telemetry=False))

    # Method to handle queries
    def ask(self, query):
        return self.qa_chain({"query": query})
