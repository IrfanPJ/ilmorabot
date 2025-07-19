import os
import re
import json
import logging
import requests
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data and State Management Classes ---

class LeadStatus(Enum):
    """Enum for lead collection status"""
    NO_INTENT = "no_intent"
    INTENT_DETECTED = "intent_detected"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_PHONE = "collecting_phone"
    COLLECTING_EMAIL = "collecting_email"
    LEAD_COMPLETE = "lead_complete"

class LeadData:
    """Class to store lead information"""
    def __init__(self):
        self.name = None
        self.phone = None
        self.email = None
        self.intent_message = None
        

# --- External Service Management ---

class AirtableManager:
    """Manages Airtable operations for lead storage"""
    def __init__(self, api_key, base_id, table_name):
        self.api_key = api_key
        self.base_id = base_id
        self.table_name = table_name
        self.base_url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._test_connection()

    def _test_connection(self):
        """Test the Airtable connection"""
        try:
            test_url = f"{self.base_url}?maxRecords=1"
            response = requests.get(test_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Airtable connection successful")
                return True
            else:
                logger.error(f"❌ Airtable connection failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Airtable connection test failed: {e}")
            return False

    def create_lead(self, lead_data):
        """Create a new lead record in Airtable"""
        try:
            fields = {}
            if lead_data.name: fields["Name"] = lead_data.name
            if lead_data.phone: fields["Phone"] = lead_data.phone
            if lead_data.email: fields["Email"] = lead_data.email
            if lead_data.intent_message: fields["Intent Message"] = lead_data.intent_message
            

            data = {"fields": fields}
            logger.info("🔄 Attempting to save lead to Airtable...")
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=10)
            
            if response.status_code == 200:
                record_id = response.json().get('id', 'Unknown')
                logger.info(f"✅ Lead successfully saved to Airtable: {lead_data.name} (ID: {record_id})")
                return True
            else:
                logger.error(f"❌ Failed to save lead to Airtable: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected error saving lead to Airtable: {e}")
            return False

# --- Core ChatBot Logic ---

class ChatBot:
    """
    A chatbot class that uses a Retrieval-Augmented Generation (RAG) pipeline
    with conversational memory, lead collection, and Airtable integration.
    """
    def __init__(self):
        """Initializes the chatbot and all its components."""
        # --- 1. Load Environment Variables ---
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.airtable_api_key = os.getenv('AIRTABLE_API_KEY')
        self.airtable_base_id = os.getenv('AIRTABLE_BASE_ID')
        self.airtable_table_name = os.getenv('AIRTABLE_TABLE_NAME', 'Leads')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')

        if not all([self.openai_api_key, self.pinecone_api_key]):
            raise ValueError("OpenAI and Pinecone API keys are missing from .env file.")
        
        self._initialize_airtable()

        # --- 2. Lead Collection State ---
        self.lead_status = LeadStatus.NO_INTENT
        self.current_lead = LeadData()

        # --- 3. Intent Keywords ---
        self.intent_keywords = [
            "i want", "i need", "help me", "can you help", "i'm interested", 
            "interested in", "tell me more", "how can i", "i would like", "sign up", 
            "get started", "my business", "our company", "we need", "looking for", 
            "implement", "solution", "consultation", "demo", "proposal", "quote", 
            "pricing", "cost", "ai solution", "machine learning", "automation", 
            "custom ai", "ai development", "ai implementation", "ai training", 
            "ai consulting", "contact", "reach out", "schedule", "meeting", "call", 
            "discuss", "talk"
        ]

        # --- 4. Initialize LangChain Components ---
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.openai_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.7,
            
        )
        
        # --- 5. Setup Vector Store and RAG Chain ---
        docsearch = self._initialize_pinecone()
        self.rag_chain = self._create_rag_chain(docsearch)
        
        print("✅ ILMORA AI Assistant with Lead Collection is ready!")

    def _initialize_airtable(self):
        """Initializes the Airtable manager if credentials are present."""
        if not all([self.airtable_api_key, self.airtable_base_id]):
            logger.warning("⚠️ Airtable credentials not found. Leads will be logged locally only.")
            self.airtable_manager = None
        else:
            logger.info("🔄 Initializing Airtable connection...")
            try:
                self.airtable_manager = AirtableManager(self.airtable_api_key, self.airtable_base_id, self.airtable_table_name)
            except Exception as e:
                logger.error(f"❌ Failed to initialize Airtable manager: {e}")
                self.airtable_manager = None

    def _initialize_pinecone(self):
        """Connects to Pinecone and sets up the vector index."""
        self.index_name = "ilmorabot"
        try:
            pc = PineconeClient(api_key=self.pinecone_api_key)
            if self.index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=self.index_name, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                self._populate_pinecone_index()
            
            return PineconeVectorStore.from_existing_index(self.index_name, self.embeddings)
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    def _populate_pinecone_index(self):
        """Loads data from a file and embeds it into the Pinecone index."""
        print("Index is empty. Loading and embedding documents...")
        file_path = './materials/ilmoradata.txt'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        print(f"Splitting document into {len(docs)} chunks.")
        PineconeVectorStore.from_documents(docs, self.embeddings, index_name=self.index_name)
        print(f"✅ Documents successfully added to index.")

    def _create_rag_chain(self, docsearch):
        """Creates the ConversationalRetrievalChain with a custom prompt and memory."""
        template = """
        You are the ILMORA AI Assistant, a specialized AI for ILMORA. Your primary goal is to answer questions while identifying and capturing business leads.
        
        ## YOUR ROLE & GUIDELINES
        1.  **Be a Knowledgeable Expert**: Use the **Retrieved Context** to answer questions about ILMORA's services. Be warm, professional, and engaging.
        2.  **Be an Astute Lead Identifier**: Listen for "buying signals" or "intent keywords" (e.g., `i need`, `solution`, `pricing`, `demo`). If you detect intent, your priority shifts to lead collection.
        3.  **Be a Smooth Lead Collector**: When intent is detected, pivot the conversation to get the user's **Name**, **Phone Number**, and **Email**. Ask for one piece of information at a time.
        
        ## LEAD COLLECTION FLOW
        - **Detect Intent**: Acknowledge their need. Example: "That's a great question, we can definitely help with that."
        - **Pivot & Ask**: Transition to lead capture. Example: "To connect you with a specialist, could I start with your name?"
        - **Confirm & Finalize**: After getting all details, confirm them and set expectations. Example: "Excellent, [Name]! I've securely logged your information. A member of our team will reach out within 24 hours."

        ## CONTEXT
        Chat History: {chat_history}
        Retrieved Context: {context}
        Human Question: {question}
        AI Response:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
            memory=memory, 
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    # --- Data Extraction Helpers ---

    def _detect_intent(self, question: str) -> bool:
        """Detects if the user shows interest in services."""
        return any(keyword in question.lower() for keyword in self.intent_keywords)

    def _extract_name(self, text: str) -> str | None:
        """Extracts a name from user input."""
        patterns = [
            r"i'?m\s+([a-zA-Z\s]+)", r"my name is\s+([a-zA-Z\s]+)",
            r"this is\s+([a-zA-Z\s]+)", r"call me\s+([a-zA-Z\s]+)",
            r"i am\s+([a-zA-Z\s]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                name = match.group(1).strip().title()
                if name and len(name.split()) <= 3 and name.lower() not in ['sure', 'yes', 'no', 'ok', 'interested']:
                    return name
        words = text.strip().split()
        if 1 <= len(words) <= 3 and all(word.isalpha() for word in words):
            name = text.strip().title()
            if name.lower() not in ['sure', 'yes', 'no', 'ok', 'interested']:
                return name
        return None

    def _extract_phone(self, text: str) -> str | None:
        """Extracts a phone number from user input."""
        pattern = r"(\+?\d{1,3}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})"
        match = re.search(pattern, text)
        return match.group(0).strip() if match else None

    def _extract_email(self, text: str) -> str | None:
        """Extracts an email address from user input."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    # --- Core State Machine ---

    def _handle_lead_collection(self, question: str) -> str | None:
        """
        Manages the lead collection state machine. This function takes priority
        over the RAG chain when the bot is in a lead collection state.
        """
        if self.lead_status == LeadStatus.LEAD_COMPLETE:
            if self._detect_intent(question):
                self.lead_status = LeadStatus.NO_INTENT
                self.current_lead = LeadData()
            else:
                return None

        if self.lead_status == LeadStatus.NO_INTENT:
            if self._detect_intent(question):
                logger.info(f"🎯 Intent detected: {question}")
                self.current_lead.intent_message = question
                self.lead_status = LeadStatus.INTENT_DETECTED
            else:
                return None

        if self.lead_status == LeadStatus.INTENT_DETECTED:
            # 1. Generate the content answer using the RAG chain
            logger.info(f"Answering the original intent question with RAG: '{self.current_lead.intent_message}'")
            rag_result = self.rag_chain.invoke({"question": self.current_lead.intent_message})
            content_answer = rag_result.get('answer', "I can certainly help with that.")

            # 2. Prepare the lead collection prompt
            lead_prompt = "I can also connect you with our team for more details. To get started, could you please share your full name?"

            # 3. Transition to the next state for the user's *next* message
            self.lead_status = LeadStatus.COLLECTING_NAME

            # 4. Return the combined response
            return f"{content_answer}\n\n{lead_prompt}"

        if self.lead_status == LeadStatus.COLLECTING_NAME:
            name = self._extract_name(question)
            if name:
                self.current_lead.name = name
                self.lead_status = LeadStatus.COLLECTING_PHONE
                return f"Thank you, {name}! What is the best phone number to reach you at?"
            else:
                return "I'm sorry, I didn't catch a valid name. Could you please provide your full name?"

        if self.lead_status == LeadStatus.COLLECTING_PHONE:
            phone = self._extract_phone(question)
            if phone:
                self.current_lead.phone = phone
                self.lead_status = LeadStatus.COLLECTING_EMAIL
                return "Perfect! And finally, what's your email address?"
            else:
                return "I didn't recognize that as a valid phone number. Could you please provide it again?"

        if self.lead_status == LeadStatus.COLLECTING_EMAIL:
            email = self._extract_email(question)
            if email:
                self.current_lead.email = email
                self._save_lead_to_airtable()
                self.lead_status = LeadStatus.LEAD_COMPLETE
                name = self.current_lead.name
                return (
                    f"Excellent, {name}! I have securely logged your information.\n\n"
                    f"A member of our team will reach out to you within 24 hours. "
                    f"Thank you for your interest in ILMORA!"
                )
            else:
                return "I couldn't find a valid email in that message. Could you please provide your email address?"
        
        return None

    def _save_lead_to_airtable(self):
        """Saves the collected lead data to Airtable or logs it locally."""
        if not self.airtable_manager:
            logger.warning("Airtable not configured. Logging lead locally.")
            logger.info(f"📝 LOCAL LEAD: Name={self.current_lead.name}, Phone={self.current_lead.phone}, Email={self.current_lead.email}")
            return
        
        if self.current_lead.name:
            
            self.airtable_manager.create_lead(self.current_lead)
        else:
            logger.error("Cannot save lead: No name was provided.")

    def ask(self, question: str) -> str:
        """Processes the user's question through the lead collection or RAG pipeline."""
        if not question.strip():
            return "Please ask a question."
        try:
            lead_response = self._handle_lead_collection(question)
            if lead_response:
                return lead_response

            logger.info(f"Processing question with RAG chain: {question}")
            result = self.rag_chain.invoke({"question": question})
            return result.get('answer', "I'm sorry, I encountered an issue. Please try again.")
            
        except Exception as e:
            logger.error(f"Critical error in ask method: {e}")
            return "I apologize, but I've encountered an unexpected error. Please try again later."
    
    def get_lead_status(self):
        """Returns the current lead collection status for debugging."""
        return {
            "status": self.lead_status.value,
            "lead_data": {
                "name": self.current_lead.name,
                "phone": self.current_lead.phone,
                "email": self.current_lead.email,
                "intent_message": self.current_lead.intent_message
            }
        }

# --- Main Execution Block ---

if __name__ == "__main__":
    try:
        chatbot = ChatBot()
        print("\n--- ILMORA AI Assistant ---")
        print("Type 'exit' to end, 'debug' for history, or 'leads' for status.\n")
      
        while True:
            user_question = input("Your Question: ")
            
            if user_question.lower() == 'exit':
                print("Thank you for chatting with ILMORA. Have a great day!")
                break
            elif user_question.lower() == 'debug':
                history = chatbot.rag_chain.memory.chat_memory.messages
                print(f"\n--- Conversation History ({len(history)} messages) ---")
                for i, msg in enumerate(history):
                    print(f"{i+1}. {msg.type}: {msg.content[:100]}...")
                continue
            elif user_question.lower() == 'leads':
                status = chatbot.get_lead_status()
                print("\n--- Lead Collection Status ---")
                print(json.dumps(status, indent=2))
                continue
            
            answer = chatbot.ask(user_question)
            print(f"\nILMORA AI: {answer}\n")

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
        logger.error(f"Critical error in main execution: {e}")