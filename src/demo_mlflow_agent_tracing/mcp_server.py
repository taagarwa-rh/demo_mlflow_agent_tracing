import re
from datetime import datetime
from typing import Union

import frontmatter
from fastmcp import FastMCP

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


def extract_markdown_content(content: str) -> Union[str, None]:
    """Extract markdown content from the given string."""
    pattern = r".*?```(markdown)?(.*?)```.*?"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(2).strip()
    return None


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
    try:
        # Create wiki page
        llm = get_chat_model()
        llm.temperature = 0.3
        prompt = WIKI_WRITER_PROMPT.format(topic=topic, date=datetime.now().date())
        retry_count = 0
        wiki_page = None
        while retry_count < 3:
            response = llm.invoke([{"role": "user", "content": prompt}])
            raw_page_content = response.content
            markdown_content = extract_markdown_content(raw_page_content)
            if markdown_content is not None:
                break
            retry_count += 1

        if wiki_page is None:
            wiki_page = raw_page_content

        # Save wiki page
        path = WIKI_PATH / (topic.title() + ".md")
        with open(path, "w") as f:
            f.write(wiki_page.strip())

        return f"Successfully wrote and saved a new wiki page for topic '{topic}'"
    except Exception as e:
        return f"Failed to write and save a new wiki page for topic '{topic}': {str(e)}"


@mcp.tool
def list_wiki_pages():
    """
    List available wiki pages.

    Returns:
        list[str]: Available wiki pages

    """  # noqa: D406
    try:
        # Get pages
        pages = list(WIKI_PATH.glob("*.md"))

        # Get page titles
        titles = [frontmatter.loads(page.read_text()).get("title") for page in pages]
        titles = [title for title in titles if title is not None]

        return titles
    except Exception as e:
        return f"Failed to list files: {str(e)}"


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
