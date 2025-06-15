import os
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from PIL import Image
import base64, io
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from transformers import AutoModel
from typing import List
import logging
from io import StringIO
from itertools import zip_longest

# Configure logging
def configure_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create a logger specifically for your ingestion pipeline
    ingestion_logger = logging.getLogger("ingestion")
    ingestion_logger.setLevel(logging.INFO)  # Set desired level

    # Prevent logs from propagating to the root logger
    ingestion_logger.propagate = False

    # Create a file handler for this logger
    file_handler = logging.FileHandler("logs/ingestion.log")
    file_handler.setLevel(logging.INFO)

    # Add console output too
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define custom log format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach the handler to your logger
    ingestion_logger.addHandler(file_handler)
    ingestion_logger.addHandler(console_handler)

    return ingestion_logger

class JinaCLIPLangchainWrapper:
    def __init__(self, model, truncate_dim=512):
        self.model = model
        self.truncate_dim = truncate_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode_text(texts, truncate_dim=self.truncate_dim)
        return embeddings.tolist()  # Ensure it returns a list of lists
    
    def embed_image(self, uris: List[str]) -> List[List[float]]:
        # images can be file paths, PIL.Image.Image, or dataURIs
        embeddings = self.model.encode_image(uris, truncate_dim=self.truncate_dim)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode_text([text], truncate_dim=self.truncate_dim)[0]
        return embedding.tolist()  # Ensure it returns a list

    
class Ingestion:
    def __init__(self):
        self.documents = []
        self.images = []
        self.image_elements = []
        self.tables = []
        self.table_texts = []
        self.captions = []

        # Initialize the embedding model
        try:
            raw_model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
        except Exception as e:
            raise Exception(f"Failed to load JinaCLIP model: {e}")
        self.embedding_model = JinaCLIPLangchainWrapper(raw_model)

        self.embeddings = [] # Initialize text embeddings
        self.image_embeddings = []  # Initialize image embeddings
        self.table_embeddings = []
         
        # Initialize the vector store
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db",  # Where to save data locally
        )

        # Configure logging
        self.ingestion_logger = configure_logging()

    def load(self, path: str, output_file: str = "document_content.txt"):
        """
        Load and parse a PDF file into structured elements (text, tables, images).
        Saves extracted tables as CSV and images as image files.
        Populates self.documents, self.tables, self.images.
        """
        
        # Parse the PDF file into structured elements (text, tables, images) using unstructured
        elements = partition_pdf(
            filename=path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            skip_infer_table_types=False,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True  # Embed Base64 image in metadata
        )

        self.ingestion_logger.info(f"[load] Loaded {len(elements)} documents from {path}")

        images = [el for el in elements if el.category == "Image"]
        self.image_elements = images  # Store raw image elements for later use
        tables = [el for el in elements if el.category == "Table"]
        self.captions = [el for el in elements if el.category == "FigureCaption"]

        # create tables, images, and content directories if they don't exist
        os.makedirs("tables", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        os.makedirs("content", exist_ok=True)

        documents = []

        for i, el in enumerate(elements):          
            if el in images:
                b64 = getattr(el.metadata, "image_base64", None)
                mime = getattr(el.metadata, "image_mime_type", None)

                # Decode and save the image from base64
                if b64:
                    try:
                        img = Image.open(io.BytesIO(base64.b64decode(b64)))
                        img.save(f"images/{i+1}.{mime.split('/')[-1]}")
                    except Exception as e:
                        self.ingestion_logger.warning(f"[load] Could not save image {el.element_id}: {e}")
                else:
                    self.ingestion_logger.error(f"[load] ❗ No image_base64 in metadata for element {el.element_id}")
            elif el in tables:
                html = getattr(el.metadata, "text_as_html", None)
                if not html:
                    self.ingestion_logger.warning(f"[load] ⚠️ Skipping element {el.element_id}: no HTML representation.")
                    continue

                try:
                    df = pd.read_html(StringIO(html))[0]
                    df.to_csv(f"tables/Table_{i+1}.csv", index=False)
                except ValueError as e:
                    self.ingestion_logger.error(f"[load] ❌ Failed to parse HTML for {el.element_id}: {e}")

                # Add the table to documents
                content = getattr(el, "text", "")
                metadata = el.metadata.to_dict() if hasattr(el, "metadata") else {}
                metadata["category"] = el.category
                documents.append(Document(page_content=content, metadata=metadata))
            else:
                content = getattr(el, "text", "")
                metadata = el.metadata.to_dict() if hasattr(el, "metadata") else {}
                metadata["category"] = el.category
                documents.append(Document(page_content=content, metadata=metadata))

        # initialize self.images as a list of image file paths stored in images folder
        pdf_images = [
            f"images/{img}" for img in os.listdir("images") if img.endswith(('.png', '.jpg', '.jpeg', '.gif'))
        ]
        self.images.extend(pdf_images)  # Store image file paths for later use

        # initialize self.tables as a list of table file paths stored in tables folder
        pdf_tables = [
            f"tables/{tbl}" for tbl in os.listdir("tables") if tbl.endswith(".csv")
        ]
        self.tables.extend(pdf_tables)  # Store table file paths for later use
        
        table_texts = []
        for table_path in self.tables:
            try:
                df = pd.read_csv(table_path)
                table_texts.append(df.to_string(index=False))
            except Exception as e:
                self.ingestion_logger.error(f"[load] ⚠️ Failed to load table {table_path}: {e}")

        self.table_texts.extend(table_texts)  # Store table texts for later use

        pdf_basename = os.path.basename(path)                 # e.g., "File_1.pdf"
        pdf_name = os.path.splitext(pdf_basename)[0]          # now just the name without extension
        file_path = os.path.join("content", pdf_name + "_" + output_file)
        self.ingestion_logger.info(f"[load] Saving content to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"[load] Loaded {len(documents)} documents from {path}\n\n")
            
            for doc in documents:
                f.write("Metadata:\n" + doc.metadata.__str__() + "\n")
                f.write("Content:\n" + doc.page_content + "\n")
                f.write("\n" + "="*80 + "\n\n")

        self.documents.extend(documents)  # Add all documents to the main list

    def load_dir(self, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                path = os.path.join(directory, filename)
                self.load(path)

    def chunk(self, chunk_size=1000, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        simplified_docs = []
        for doc in self.documents:
            metadata = doc.metadata
            # Select only simple, relevant fields
            simple_metadata = {
                k: metadata[k]
                for k in ["filetype", "page_number", "file_directory", "filename", "category"]
                if k in metadata
            }

            chunks = splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[simple_metadata]  # Now included safely
            )
            simplified_docs.extend(chunks)

        self.documents = simplified_docs

        self.ingestion_logger.info(f"[chunk] Split all documents into {len(self.documents)} chunks")


    def embed(self):          
        # Encode text and images
        self.ingestion_logger.info("[embed] Generating embeddings for documents...")
        texts = [doc.page_content for doc in self.documents]
        text_embeddings = self.embedding_model.embed_documents(texts)
        
        self.ingestion_logger.info("[embed] Generating embeddings for images...")
        image_embeddings = self.embedding_model.embed_images(self.images)

        self.ingestion_logger.info("[embed] Generating embeddings for tables...")
        table_embeddings = self.embedding_model.embed_documents(self.table_texts)
        self.table_embeddings = table_embeddings      
        
        self.embeddings = text_embeddings # Store text embeddings
        self.image_embeddings = image_embeddings  # Store image embeddings
        self.table_embeddings = table_embeddings  # Store table embeddings
        
        self.ingestion_logger.info(f"[embed] Generated {len(text_embeddings)} text embeddings")
        self.ingestion_logger.info(f"[embed] Generated {len(image_embeddings)} image embeddings")
        self.ingestion_logger.info(f"[embed] Generated {len(table_embeddings)} table embeddings")

    def store(self):
        docs = self.documents        
        self.ingestion_logger.info(f"[store] Storing {len(docs)} chunks with embeddings into Chroma")

        # Insert documents into Chroma vector store
        self.vector_store.add_documents(
            documents=docs
        )
        
        self.ingestion_logger.info(f"[store] Stored {len(docs)} vectors into Chroma vector store")

        if len(self.images) > 0:
            self.ingestion_logger.info(f"[store] Storing {len(self.images)} images into Chroma")

            captions = []
            for img in self.image_elements:
                # Find closest caption (e.g., same page & element before it)
                caption_text = next(
                    (cap.text for cap in self.captions if cap.metadata.page_number == img.metadata.page_number),
                    ""
                )
                captions.append(caption_text)
            
            pairs = zip_longest(self.images, captions, fillvalue="")

            image_metadatas = []
            image_ids = []

            for i, (img_path, caption) in enumerate(pairs):
                if img_path is None:
                    continue  # No image to store
                md = {
                    "category": "Image",
                    "source": img_path,
                    "image_index": i,
                    "caption": caption
                }
                image_metadatas.append(md)
                image_ids.append(f"img_{i}")

            self.vector_store.add_images(
                uris=[md["source"] for md in image_metadatas],
                metadatas=image_metadatas,
                ids=image_ids
)

def main():
    
    ing = Ingestion()

    # ing.load("../../data/Siamese Neural Networks for One-shot Image Recognition.pdf")
    ing.load_dir("../../data")
    ing.chunk(chunk_size=1000, chunk_overlap=200)
    # ing.embed()
    ing.store()

if __name__ == "__main__":
    main()