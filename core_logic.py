#
# -------------------- core_logic.py (Complete Version) --------------------
#
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate
from supabase.client import Client, create_client
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory

load_dotenv()

# --- Tool Creation Functions ---
def create_graph_qa_tool():
    graph = Neo4jGraph()
    chain = GraphCypherQAChain.from_llm(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
        graph=graph, verbose=False, allow_dangerous_requests=True
    )
    return Tool(
        name="Knowledge Graph Search", func=chain.invoke,
        description="Use for specific questions about rules, policies, costs, fees. e.g., 'What is the cakeage fee?'"
    )

def create_vector_search_tool():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase URL or Service Key is missing. Please check your .env file.")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = SupabaseVectorStore(
        client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents"
    )
    retriever = vector_store.as_retriever()
    return Tool(
        name="General Information Search", func=retriever.invoke,
        description="Use for general, conceptual, or 'how-to' questions, and for information like operating hours and location. e.g., 'How do I prepare for a party?' or 'What are your hours?'"
    )

# --- Tool Schemas and Functions ---

class BookGymTrialArgs(BaseModel):
    name: str = Field(description="The user's full name.")
    email: str = Field(description="The user's email address.")
    phone: str = Field(description="The user's phone number.")

def book_gym_trial(name: str, email: str, phone: str) -> str:
    """Books a 7-day free gym trial."""
    print(f"--- ACTION: Sending lead to sales team ---")
    print(f"Name: {name}, Email: {email}, Phone: {phone}")
    print(f"--- END ACTION ---")
    return f"Great news, {name}! I've scheduled your 7-day free trial. A sales representative will contact you shortly at {phone} or {email} to confirm the details. Get ready to have a great workout! 潮"

class GatherPartyDetailsArgs(BaseModel):
    num_kids: int = Field(description="The number of children attending the party.")
    age_range: str = Field(description="The approximate age range of the children.")
    desired_date: str = Field(description="The user's desired date for the party.")

def gather_party_details(num_kids: int, age_range: str, desired_date: str) -> str:
    """Gathers initial details for a kids' party inquiry."""
    print(f"--- ACTION: Party lead gathered ---")
    print(f"Kids: {num_kids}, Ages: {age_range}, Date: {desired_date}")
    print(f"--- END ACTION ---")
    cost_per_child = 350
    estimated_cost = num_kids * cost_per_child
    return f"Awesome! For a party of {num_kids} kids around the age of {age_range} on {desired_date}, you're looking at an estimated cost of R{estimated_cost}. I've sent these details to our party coordinators, and they'll be in touch soon to help plan the perfect celebration! 肢"

class EscalateToHumanArgs(BaseModel):
    name: str = Field(description="The user's full name.")
    phone: str = Field(description="The user's phone number.")
    reason: str = Field(description="A brief reason for the user's request to speak to a human.")

def escalate_to_human(name: str, phone: str, reason: str) -> str:
    """Handles a user's request to speak to a person."""
    print(f"--- ACTION: Escalating to human support ---")
    print(f"Name: {name}, Phone: {phone}, Reason: {reason}")
    print(f"--- END ACTION ---")
    return f"Thank you, {name}. I've passed your request on to our team. Someone will call you back at {phone} as soon as possible to help with: '{reason}'. "


# --- Main Agent Initialization Function ---
def initialize_agent(memory):
    """Creates and returns the main agent executor, now with conversation memory."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
    
    tools = [
        create_graph_qa_tool(),
        create_vector_search_tool(),
        StructuredTool.from_function(
            name="Book 7-Day Gym Trial", func=book_gym_trial, args_schema=BookGymTrialArgs,
            description="Use this when a user wants to book, schedule, or start a 7-day free trial for the IPIC Active gym. You must ask for their full name, email, and phone number first before using this tool."
        ),
        StructuredTool.from_function(
            name="Gather Party Inquiry Details", func=gather_party_details, args_schema=GatherPartyDetailsArgs,
            description="Use this tool when a user wants to inquire about booking a kids' party and needs a price estimate. You must ask for the number of kids, their age range, and the desired date first."
        ),
        StructuredTool.from_function(
            name="Escalate to a Human", func=escalate_to_human, args_schema=EscalateToHumanArgs,
            description="Use this tool when the user explicitly asks to speak to a person, staff member, or human. You must ask for their name, phone number, and a brief reason for their request first."
        )
    ]

    # The persona template is now defined here, using this improved version
    persona_template = """
    You are a helpful assistant for IPIC Active (a gym) and IPIC Play (a kids' play park).
    Your name is "Sparky," the friendly and energetic guide for our family hub.

    **Your Persona:**
    - **Friendly & Professional:** Be warm, welcoming, and clear in your answers.
    - **Playful Energy:** Use emojis where appropriate.
    - **Tool Usage:** Your primary purpose is to use the provided tools to answer user questions. Do not make up information.
    - **Handling Non-Questions:** If the user's input is not a clear question or command (e.g., they just say "ok", "thanks", or send a greeting again), you do not need to use a tool. In this case, your thought process should conclude that you can answer directly.
    - **Conversation Flow:** Remember the conversation history. Only greet the user once.

    **You have access to the following tools:**
    {tools}

    **Use the following format:**

    Question: the input question you must answer
    Thought: You must think about what to do. I need to analyze the user's 'New question' and the 'Previous conversation history'.
    1. Is this a clear question that one of my tools can answer? If yes, I will choose the best tool from [{tool_names}] and prepare the Action Input.
    2. Is this a simple greeting, a thank you, or a phrase that doesn't need a tool? If yes, I have enough information to respond directly without using a tool.
    After this analysis, I will either use a tool or proceed directly to the Final Answer.
    Action: The action to take, should be one of [{tool_names}].
    Action Input: The input to the action.
    Observation: The result of the action.
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now have enough information to answer the user in my Sparky persona.
    Final Answer: Your final, customer-facing answer.

    Begin!

    Previous conversation history:
    {history}

    New question: {input}
    Thought:{agent_scratchpad}
    """

    # Correctly create the prompt template with placeholders
    prompt = PromptTemplate.from_template(persona_template)

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True, max_iterations=7)