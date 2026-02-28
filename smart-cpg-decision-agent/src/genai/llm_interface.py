import os

def get_llm():
    """
    Returns an instance of an LLM based on environment variables.
    Supports HuggingFace and Google Gemini (Google AI Studio).
    Defaults to a simple mock or HuggingFace if keys are missing.
    """
    # Check for Google Gemini (Google AI Studio)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        except ImportError:
            print("Please install langchain-google-genai to use Google Gemini.")

    # Check for HuggingFace Hub
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
            )
            return ChatHuggingFace(llm=llm)
        except ImportError:
            print("Please install langchain-huggingface to use HuggingFace Hub.")

    # Check for standard OpenAI (fallback)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except ImportError:
            print("Please install langchain-openai to use OpenAI.")

    # If no keys are found, raise an error or return a mock
    raise ValueError("No valid LLM API key found in environment variables. Please set GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN, or OPENAI_API_KEY.")
