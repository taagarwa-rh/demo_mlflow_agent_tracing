import re
from datetime import datetime

import frontmatter
from fastmcp import FastMCP
from langchain_core.tools import ToolException

from demo_mlflow_agent_tracing.chat_model import get_chat_model
from demo_mlflow_agent_tracing.constants import WIKI_PATH

mcp = FastMCP("Content Writer")

WIKI_WRITER_PROMPT = """You are an expert wiki writer. Today's date is {date}.

Write a medium-sized wiki page in markdown about "{topic}". Your article must satisfy the following requirements.

- It must include Markdown frontmatter in `---` format, containing a 'title', 'description', and 'created_date' metadata.
- You must include a title
- Use paragraphs and section headers to clearly organize content
- Use topic appropriate headers, for example:
    - Historical Figure: Early Life, Career, Personal Life, plus subheaders for relevant achievements / topics
    - Technology: History, Usage, plus subheaders for relevant topics
    - Animal: Description, Behavior, Range, plus subheaders for relevant topics
    - and so on...
- Limit the use of bullet points, focus on writing meaningful paragraphs where possible
    
Wrap your response in backticks like so:

```markdown
---
title: "Page Title"
description: "Description of the page"
created_date: "{date}"
---

# Page Title

...your markdown content...

```
""".strip()


@mcp.tool
def create_new_wiki_page(topic: str) -> str:
    """
    Create and save a new wiki page.

    This will write a full article about the given topic, then save it to the file system.

    Args:
        topic (str): The user's requested topic.

    Returns:
        str: Success message when wiki page creation is successful.
    """
    # Create wiki page
    llm = get_chat_model()
    llm.temperature = 0.3
    prompt = WIKI_WRITER_PROMPT.format(topic=topic, date=datetime.now().date())
    pattern = r"```markdown(.*)```"
    retry_count = 0
    wiki_page = None
    while retry_count < 3:
        response = llm.invoke([{"role": "user", "content": prompt}])
        raw_page_content = response.content
        if re.match(pattern, raw_page_content, flags=re.DOTALL | re.MULTILINE):
            wiki_page = re.search(pattern, raw_page_content, flags=re.DOTALL | re.MULTILINE).group(1)
            break
        retry_count += 1

    if wiki_page is None:
        raise ToolException("Failed to generate wiki page. Please try again later.")

    # Save wiki page
    path = WIKI_PATH / (topic.title() + ".md")
    with open(path, "w") as f:
        f.write(wiki_page.strip())

    return f"Successfully wrote and saved a new wiki page for topic '{topic}'"


@mcp.tool
def list_wiki_pages():
    """
    List available wiki pages.

    Returns:
        list[str]: Available wiki pages

    """  # noqa: D406
    # Get pages
    pages = list(WIKI_PATH.glob("*.md"))

    # Get page titles
    titles = [frontmatter.loads(page.read_text()).get("title") for page in pages]
    titles = [title for title in titles if title is not None]

    return titles


# TODO
# @mcp.tool
# def search_wiki_pages(query: str):
#     """
#     Search the content of wiki pages for specific information.

#     Uses a similarity based search, not keywords.

#     Args:
#         query (str): The query to search for.

#     Returns:
#         list[str]: Chunks from wiki pages that match the query.
#     """


if __name__ == "__main__":
    mcp.run(show_banner=False)
