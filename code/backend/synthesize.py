from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from retrieve import Retrieval
from logger import logger
import os
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH


class Synthesis:
    def __init__(self, groq_api_key: str):
        # Load application configurations
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        llm = app_config["llm"]

        self.llm = ChatGroq(api_key=groq_api_key, model_name=llm)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # These are initialized later
        self.chain = None
        self.retriever = None
        self.retrieved_docs = []

    def configure_prompt_settings(self, relevant_docs: list, retrieval: Retrieval):
        # Load prompt configurations
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

        rag_assistant_prompt = prompt_config["rag_assistant_prompt"]
        
        self.retriever = retrieval.get_vector_store().as_retriever()
        self.retrieved_docs = relevant_docs

        # Custom system prompt
        configured_prompt = build_prompt_from_config(
            config=rag_assistant_prompt,
            input_data=relevant_docs if relevant_docs else "",
        )

        # Remove '{' and '}' from configured prompt - required for LangChain to not confuse it with input variables
        configured_prompt = configured_prompt.replace("{", "").replace("}", "")

        base_template = """
        Instructions:
        {instructions}

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {question}

        Answer:
        """

        prompt_template = PromptTemplate(
            template=base_template,
            input_variables=["context", "chat_history", "question", "instructions"]
        )

        custom_prompt = prompt_template.partial(instructions=configured_prompt)
        
        # Log the custom prompt for debugging
        logger.info(f"[Synthesis] Custom prompt: {custom_prompt}")

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose=True
        )

    def get_llm_response(self, query: str) -> tuple[str, list[str], list[str]]:
        """Get the LLM response for a given query."""
        logger.info(f"[Synthesis] Getting LLM response for query: {query}")

        result = self.chain.invoke({"question": query})
        answer = result["answer"]

        image_paths = []
        table_paths = []

        for doc in self.retrieved_docs:
            metadata = doc.metadata
            if metadata.get("category") == "Image":
                image_paths.append(f"/images/{os.path.basename(metadata['filename'])}")
            elif metadata.get("category") == "Table":
                filename = metadata.get("filename")
                if filename:
                    table_paths.append(f"/tables/{filename}")

        return answer, image_paths, table_paths