from langchain.memory import ConversationBufferMemory

def get_memory(memory_key="chat_history"):
    """
    Returns a standard ConversationBufferMemory for the agent loop.
    This maintains the context of the conversation.
    Explicitly define the output_key to avoid errors during standard dict extraction.
    """
    return ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True,
        output_key="output"  # explicitly set this for standard Tool Calling Agent Executor outputs
    )
