from langchain_core.tools import Tool, tool
import wikipedia
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

search = GoogleSerperAPIWrapper()

serper_tool = Tool(
    func=search.run,
    name="Google_Serper_search",
    description="Search Google for real-time info",
)

@tool
def wikipedia_tool(page: str, detail: str = "summary") -> str:
    """Get a Wikipedia page. Useful for retrieving factual information.
    
    args:
        page: The title of the Wikipedia page to retrieve.
        detail: "summary" for summary, "full" for full content. Use summary when simple info is needed.
    
    """

    # print("Wikipedia tool working...")

    if detail == "full":
        return wikipedia.page(page, auto_suggest=False).content
    else:
        return wikipedia.page(page, auto_suggest=False).summary
    
tools = [serper_tool, wikipedia_tool]