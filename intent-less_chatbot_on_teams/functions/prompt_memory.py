from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

def prompt_memory(memory_type, llm, max_conversation_turns=None, max_token_limit=None, memory_file_path=None):
    if memory_type == 'ConversationSummaryBufferMemory':
        with open(memory_file_path, 'r', encoding='utf-8') as file:
            memory_prompt_text = file.read()
            memory_prompt = PromptTemplate(input_variables=['summary', 'new_lines'], template=memory_prompt_text)
            memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit, prompt=memory_prompt)

    elif memory_type == 'ConversationBufferWindowMemory':
        memory = ConversationBufferWindowMemory(k=max_conversation_turns)

    elif memory_type == 'ConversationBufferMemory':
        memory = ConversationBufferMemory()

    else:
        raise ValueError(f"Invalid memory type: {memory_type}")

    return memory