import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import logging

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    """
    A chatbot class that uses a Retrieval-Augmented Generation (RAG) pipeline
    with OpenAI and Pinecone to answer questions based on a provided document,
    now with conversational memory and improved error handling.
    """
    def __init__(self):
        """
        Initializes the chatbot by setting up environment variables,
        connecting to Pinecone, loading and processing documents if needed,
        and creating the conversational RAG chain.
        """
        # --- 1. Load Environment Variables ---
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')

        if not all([self.openai_api_key, self.pinecone_api_key]):
            raise ValueError("API keys not found in .env file. Please check your configuration.")

        # --- 2. Initialize Embeddings and LLM ---
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        # FIX: Use correct model name
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # Fixed: was "gpt-4.1-nano" which doesn't exist
            openai_api_key=self.openai_api_key,
            temperature=0.7  # Added some creativity for better conversations
        )
        
        # --- 3. Setup Pinecone Vector Store ---
        self.index_name = "ilmorabot"
        docsearch = self._initialize_pinecone()

        # --- 4. IMPROVED PROMPT ---
        template = """
You are the ILMORA AI Assistant, a helpful and knowledgeable AI representing ILMORA - a leading organization dedicated to democratizing artificial intelligence knowledge and delivering transformative solutions across industries.

## ABOUT ILMORA
ILMORA's mission is to power businesses with custom AI and prepare communities for intelligent futures. We work closely with businesses as trusted partners to understand their unique challenges and provide practical, impactful AI solutions through:
- Industry-specific AI solutions
- Comprehensive education programs  
- Ongoing support for sustainable growth
- Education-first, end-to-end partnership model
- Real-world AI solutions that deliver measurable results

## YOUR ROLE & CAPABILITIES
You are a conversational AI assistant that can:
✅ Handle greetings, farewells, and basic conversation naturally
✅ Remember user names and maintain context throughout the conversation
✅ Answer questions about ILMORA's services, approach, and capabilities
✅ Provide helpful guidance and next steps
✅ Be warm, professional, and engaging

## CONVERSATION GUIDELINES

**For Greetings (hi, hello, hey):**
- Respond warmly and introduce yourself as the ILMORA AI Assistant
- Briefly mention you're here to help with questions about ILMORA's AI solutions
- Be friendly and inviting

**For Farewells (bye, goodbye, see you later):**
- Thank them for their interest in ILMORA
- Offer future assistance
- End on a positive, helpful note

**For Name Introductions:**
- Acknowledge their name warmly
- Remember and use their name in future responses
- Show appreciation for the introduction

**For Questions about ILMORA:**
- Use the retrieved context to provide comprehensive, helpful answers
- Be specific and detailed when you have relevant information
- Connect their questions to ILMORA's services and capabilities
- List out every thing user is asking for from the context

**For General Questions:**
- Try to be helpful while staying focused on ILMORA
- If appropriate, relate the topic back to ILMORA's expertise
- Be honest about your limitations while offering alternatives

## RESPONSE PRINCIPLES
1. **Be Conversational**: Handle basic social interactions naturally
2. **Be Helpful**: Provide specific, actionable information when available
3. **Be Honest**: Clearly state when you don't have specific information
4. **Be Professional**: Maintain ILMORA's professional image
5. **Be Engaging**: Keep conversations interesting and informative

## FALLBACK RESPONSE
Only use this when you truly don't have relevant information:
"I don't have that specific information about ILMORA in my knowledge base. For the most accurate and detailed information about [topic], I recommend reaching out to our team directly."

## CONTEXT USAGE
Use the retrieved context below to answer questions about ILMORA. When you have relevant context, provide detailed, helpful responses.

---

Chat History:
{chat_history}

Retrieved Context:
{context}

Human Question: {question}

AI Response:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])
        
        # --- 5. Create Conversational Memory ---
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"  # Added to prevent memory issues
        )

        # --- 6. Create the Conversational RAG Chain ---
        self.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=docsearch.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 most relevant chunks
            memory=memory, 
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True  # Added for debugging
        )
        print("✅ ILMORA AI Assistant is ready!")

    def _initialize_pinecone(self):
        """
        Connects to Pinecone, creates the index if it doesn't exist,
        and either populates it with documents or loads the existing index.
        """
        try:
            pc = PineconeClient(api_key=self.pinecone_api_key)
            
            if self.index_name not in pc.list_indexes().names():
                print(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            index = pc.Index(self.index_name)
            index_stats = index.describe_index_stats()
            
            if index_stats['total_vector_count'] == 0:
                print("Index is empty. Loading and embedding documents...")
                
                # Check if file exists
                file_path = './materials/ilmoradata.txt'
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Document file not found: {file_path}")
                
                loader = TextLoader(file_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                docs = text_splitter.split_documents(documents)
                
                print(f"Splitting document into {len(docs)} chunks.")
                
                docsearch = PineconeVectorStore.from_documents(
                    docs, 
                    self.embeddings, 
                    index_name=self.index_name
                )
                print(f"✅ Documents successfully added to index. Total records: {len(docs)}")
            else:
                print("Index already populated. Loading existing index.")
                docsearch = PineconeVectorStore.from_existing_index(self.index_name, self.embeddings)
                
            return docsearch
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    def ask(self, question: str) -> str:
        """
        Asks a question to the conversational RAG chain and returns the answer.
        Now with improved error handling and logging.
        """
        if not question.strip():
            return "Please ask a question."
            
        try:
            logger.info(f"Processing question: {question}")
            result = self.rag_chain.invoke({"question": question})
            
            # Log retrieved context for debugging
            if 'source_documents' in result and result['source_documents']:
                logger.info(f"Retrieved {len(result['source_documents'])} relevant documents")
            else:
                logger.warning("No relevant documents retrieved")
            
            return result['answer']
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # More specific error handling
            if "rate limit" in str(e).lower():
                return "I'm currently experiencing high demand. Please try again in a moment."
            elif "api" in str(e).lower():
                return "I'm having trouble connecting to my knowledge base. Please try again."
            else:
                return "I apologize, but I encountered an error. Please try rephrasing your question or contact our support team."

    def get_conversation_history(self):
        """Get the current conversation history for debugging"""
        return self.rag_chain.memory.chat_memory.messages

# --- Usage Example ---
if __name__ == "__main__":
    try:
        chatbot = ChatBot()
        print("\n--- ILMORA AI Assistant ---")
        print("Type 'exit' to end the conversation.")
        print("Type 'debug' to see conversation history.\n")
     
        while True:
            user_question = input("Your Question: ")
            
            if user_question.lower() == 'exit':
                print("Thank you for chatting with ILMORA AI Assistant. Have a great day!")
                break
            elif user_question.lower() == 'debug':
                history = chatbot.get_conversation_history()
                print(f"\n--- Conversation History ({len(history)} messages) ---")
                for i, msg in enumerate(history):
                    print(f"{i+1}. {msg.type}: {msg.content[:100]}...")
                continue
            
            answer = chatbot.ask(user_question)
            print(f"ILMORA AI: {answer}")

    except ValueError as ve:
        print(f"❌ Configuration Error: {ve}")
        print("Please check your .env file and ensure all required API keys are present.")
    except FileNotFoundError as fe:
        print(f"❌ File Error: {fe}")
        print("Please ensure the ilmoradata.txt file exists in the materials/ directory.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        logger.error(f"Critical error in main: {e}")
        print("Please check your configuration and try again.")