#
# -------------------- chatbot_cli.py --------------------
#
from core_logic import initialize_agent

def get_agent_response(agent_executor, query):
    """Invokes the agent and cleans the output."""
    try:
        response = agent_executor.invoke({"input": query})
        raw_output = response["output"]
        if "Final Answer:" in raw_output:
            clean_response = raw_output.split("Final Answer:")[-1].strip()
        else:
            clean_response = raw_output.strip()
        return clean_response if clean_response else "I've processed the information, but I don't have a specific answer to provide."
    except Exception as e:
        print(f"Agent invocation error: {e}")
        return "I'm sorry, but I encountered an error while trying to process your request."

def main_cli():
    """Main function for the command-line interface."""
    print("⚙️  Setting up Sparky, the Sophisticated Agent for CLI...")
    agent_executor = initialize_agent()
    print("\n🤖 Hi, I'm Sparky! How can I help you today? ✨")
    print("=" * 50)
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        response = get_agent_response(agent_executor, query)
        print("\n🤖 Sparky:", response)

if __name__ == "__main__":
    main_cli()