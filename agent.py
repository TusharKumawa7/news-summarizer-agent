import os
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()
model_name = os.getenv("MODEL", "gemini-2.0-flash")

def save_topic_to_state(tool_context: ToolContext, topic: str) -> dict:
    """Saves the user's news topic to the agent's shared state."""
    tool_context.state["TOPIC"] = topic
    return {"status": "success", "topic": topic}

wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

researcher_agent = Agent(
    name="researcher_agent",
    model=model_name,
    description="Researches the given topic using Wikipedia.",
    instruction="""
    You are a news research assistant. Research the TOPIC thoroughly.
    Use the Wikipedia tool to find background information.
    TOPIC: { TOPIC }
    """,
    tools=[wikipedia_tool],
    output_key="research_data"
)

summarizer_agent = Agent(
    name="summarizer_agent",
    model=model_name,
    description="Summarizes research into a clear news brief.",
    instruction="""
    You are a professional news editor. Using the RESEARCH_DATA:
    ## 📰 News Summary
    ## 🔑 Key Takeaways
    ## 💡 Why It Matters
    RESEARCH_DATA: { research_data }
    """
)

news_workflow = SequentialAgent(
    name="news_workflow",
    description="Researches a topic and delivers a clean news summary.",
    sub_agents=[researcher_agent, summarizer_agent]
)

root_agent = Agent(
    name="news_summarizer",
    model=model_name,
    description="An AI news summarizer.",
    instruction="""
    You are a friendly AI News Summarizer.
    Greet the user, ask what topic they want summarized.
    Use save_topic_to_state tool, then transfer to news_workflow.
    """,
    tools=[save_topic_to_state],
    sub_agents=[news_workflow]
)
