from typing import List, Set, Optional
from collections import defaultdict

from llama_cpp import Llama
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.workflow import Context
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

def convert_message_list_to_str(messages):
    chat_history_str = ""
    for message in messages:
        if type(message["content"]) is str:
            chat_history_str += message["role"] + ": " + message["content"] + "\n"
        else:
            chat_history_str += message["role"] + ": " + message["content"]["text"] + "\n"
                
    return chat_history_str

class SetupEvent(Event):
    pass

class CondenseQueryEvent(Event):
    condensed_query_str: str

class RetrievalEvent(Event):
    nodes: List[NodeWithScore]

class RAGAgent(Workflow):
    SUMMARY_TEMPLATE = (
        "Given the chat history:\n"
        "'''{chat_history_str}'''\n\n"
        "And the user asked the following question:{query_str}\n"
        "Rewrite to a standalone question:\n"
    )

    CONTEXT_PROMPT_TEMPLATE = (
        "Information that might help:\n"
        "-----\n"
        "{node_context}\n"
        "-----\n"
        "Please write a response to the following question, using the above information if relevant:\n"
        "{query_str}\n"
    ) 
    def __init__(self, embed_model_name: str, llm: Llama, timeout: int = 60, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.k = 3
        self.search_index = None
        self.retriever = None
        self.chat_history = []
        self.lookup_files = set()

        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = self.embed_model
        self.node_processor = SimilarityPostprocessor(similarity_cutoff=0.3)
        self.llm = llm

    def udpate_index(self, files: Optional[Set[str] ] = set()):
        if not files:
            print("No lookup files provided, clearing index...")
            self.retriever = None
            self.search_index = None
            return
        
        print("Indexing documents...")
        documents = SimpleDirectoryReader(input_files=files, recursive=True).load_data(show_progress=True, num_workers=1)
        page_num_tracker = defaultdict(int)
        for doc in documents:
            key = doc.metadata['file_path']
            doc.metadata['page_index'] = page_num_tracker[key]
            page_num_tracker[key] += 1

        if self.search_index is None:
            self.search_index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
        else:
            for doc in documents:
                self.search_index.insert(doc) # Add the new document to the index

        self.retriever = self.search_index.as_retriever(similarity_top_k=self.k)
    
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> SetupEvent:
        # set frequetly used variables to context
        query_str = ev.query_str
        image = ev.image
        lookup_files = ev.lookup_files
        streaming = ev.streaming
        await ctx.set("query_str", query_str)
        await ctx.set("image", image)
        await ctx.set("streaming", streaming)

        # update index if needed
        if lookup_files != self.lookup_files:
            print("Different lookup files, updating index...")
            self.udpate_index(lookup_files)

        self.lookup_files = lookup_files.copy()

        return SetupEvent()

    @step
    async def condense_history_to_query(self, ctx: Context, ev: SetupEvent) -> CondenseQueryEvent:
        """
            Condense the chat history and the query into a single query. Only used for retrieval.
        """
        query_str = await ctx.get("query_str")

        formated_query = ""
        
        if len(self.chat_history) > 0 or self.retriever is not None:
            chat_history_str = convert_message_list_to_str(self.chat_history)
            formated_query = self.SUMMARY_TEMPLATE.format(chat_history_str=chat_history_str, query_str=query_str)
            history_summary = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": formated_query}], stream=False
            )["choices"][0]["message"]["content"]
            condensed_query = "Context:\n" + history_summary + "\nQuestion: " + query_str
        else:
            # if there is no history or no need for retrieval, return the query as is
            condensed_query = query_str

        return CondenseQueryEvent(condensed_query_str=condensed_query)
    
    @step
    async def retrieve(self, ctx: Context, ev: CondenseQueryEvent) -> RetrievalEvent:
        # retrieve from context
        if not self.retriever:
            return RetrievalEvent(nodes=[])

        condensed_query_str = ev.condensed_query_str
        nodes = await self.retriever.aretrieve(condensed_query_str)
        nodes = self.node_processor.postprocess_nodes(nodes)
        return RetrievalEvent(nodes=nodes)
    
    def _prepare_query_with_context(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
    ) -> str:
        node_context = ""

        if len(nodes) == 0:
            return query_str
        
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"\n{node_text}\n\n"

        formatted_query = self.CONTEXT_PROMPT_TEMPLATE.format(
            node_context=node_context, query_str=query_str
        )
        
        return formatted_query

    @step
    async def llm_response(self,  ctx: Context, retrieval_ev: RetrievalEvent) -> StopEvent:
        nodes = retrieval_ev.nodes
        query_str = await ctx.get("query_str")
        image = await ctx.get("image")
        query_with_ctx = self._prepare_query_with_context(query_str, nodes)
        streaming = await ctx.get("streaming", False)

        if image:
            formated_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_with_ctx},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        else:
            formated_message = {"role": "user", "content": query_with_ctx}

        response = self.llm.create_chat_completion(
            messages=self.chat_history+[formated_message], stream=streaming
        )
        self.chat_history.append({"role": "user", "content": query_str})

        return StopEvent(result=response)

    
        