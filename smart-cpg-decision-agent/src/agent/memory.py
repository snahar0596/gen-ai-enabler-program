from langchain.memory import ConversationBufferMemory

def get_memory(memory_key="chat_history"):
    """
    Returns a standard ConversationBufferMemory for the agent loop.
    This maintains the context of the conversation.
    """
    return ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True
    )
