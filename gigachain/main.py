from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI, GigaChat
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper

# LLM initalize
llm = 'СЮДА LLM'

# Searcher initalize
search = SerpAPIWrapper()

# Confog initalize
tools = [
    Tool(
        name="Промежуточный ответ",
        func=search.run,
        description="Используется, когда нужно вопспользоваться поиском",
    )
]

# Agent initalize
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)

# Run agent
self_ask_with_search.run("Как звали жену человека, который придумал психотипы")
