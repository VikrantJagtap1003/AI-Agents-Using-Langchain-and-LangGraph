from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from langgraph.store.redis import RedisStore
from pydantic import BaseModel,Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from langgraph.types import Send

load_dotenv(Path(__file__).parent.parent / ".env")

#model 
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

#Input.  ---> AI call. --> output 

class BlogTask(BaseModel):
    id:           str            = Field(...,  description="Unique ID of the task")
    name:         str            = Field(...,  description="Name of the task")
    brief:        str            = Field(...,  description="What things we need to cover")
    order:        int            = Field(...,  description="Section order in the blog", ge=1)
    word_count:   int            = Field(300,  description="Target word count", ge=100, le=2000)
    keywords:     list[str]      = Field([],   description="SEO keywords to include")
    section_type: Literal["intro", "body", "conclusion"] = Field("body")
    tone:         Literal["formal", "casual", "technical"] = Field("casual")

class Plan(BaseModel):
    title:           str        = Field(..., description="Title of the blog")
    description:     str        = Field(..., description="Description of the blog")
    tasks:           list[BlogTask] = Field(..., description="List of tasks")
    target_audience: str        = Field(..., description="Who this blog is written for")
    total_word_count:int        = Field(1500, description="Total target word count", ge=500)
    tags:            list[str]  = Field([],   description="SEO tags for the blog")
    tone:            Literal["formal", "casual", "technical"] = Field("casual")


class WorkerPayload(TypedDict):
    task:  BlogTask
    topic: str
    plan:  Plan

class BlogState(TypedDict):
    topic: str
    plan: Plan
    results:Annotated[list[dict],operator.add]
    final_blog:Annotated[str, "Final blog content after all tasks are completed"]



#####################################################################


def workerNode(state: WorkerPayload):

    task: BlogTask = state['task']
    topic: str     = state['topic']
    plan: Plan     = state['plan']

    keywords_str = ", ".join(task.keywords) if task.keywords else "none specified"

    result = model.invoke(
        [
            SystemMessage(content=(
                f"You are an expert blog writer specializing in {task.tone} content. "
                "Your job is to write one focused section of a larger blog post. "
                "Follow these rules strictly:\n"
                "- Write ONLY your assigned section — do not write other sections\n"
                "- Use markdown formatting (##, ###, bullet points, bold where needed)\n"
                "- Stay within the target word count\n"
                "- Naturally weave in any provided SEO keywords\n"
                "- Match the tone and audience of the overall blog plan\n"
                "- Be specific, avoid filler phrases like 'In conclusion' or 'It is worth noting'"
            )),
            HumanMessage(content=(
                f"Blog topic: {topic}\n"
                f"Target audience: {plan.target_audience}\n"
                f"Overall blog tone: {plan.tone}\n\n"
                f"Your section:\n"
                f"  Name: {task.name}\n"
                f"  Type: {task.section_type} (section {task.order} of {len(plan.tasks)})\n"
                f"  Brief: {task.brief}\n"
                f"  Target word count: {task.word_count} words\n"
                f"  SEO keywords to include: {keywords_str}\n\n"
                "Write this section now in markdown. Do not include a title for the whole blog — only the section heading."
            )),
        ]
    ).content

    return {"results": [{"order": task.order, "section": task.name, "content": result}]}


def orchestorator(state:BlogState):

    structured_llm_output = model.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=(
                "You are a senior content strategist and blog planner. "
                "Your job is to break a blog topic into a clear, well-scoped set of sections that a team of writers can work on independently. "
                "Follow these rules:\n"
                "- Create 4–6 focused tasks (not too broad, not too narrow)\n"
                "- Always start with an 'intro' section and end with a 'conclusion' section\n"
                "- Each task brief must be specific enough that a writer knows exactly what to cover\n"
                "- Assign realistic word counts that sum to the total_word_count\n"
                "- Include 3–5 relevant SEO keywords per task\n"
                "- Set target_audience and tone to guide all writers consistently"
            )),
            HumanMessage(content=(
                f"Create a detailed blog plan for the topic: \"{state['topic']}\"\n\n"
                "The plan should produce a high-quality, well-researched blog post "
                "that is informative, engaging, and optimized for search engines."
            )),
        ]
    )
    return {"plan": structured_llm_output}


def fanout(state:BlogState):
    tasks = state['plan'].tasks
    results = []
    for task in tasks:
        results.append(Send("workerNode",{ "task":task ,"topic":state['topic'],"plan":state['plan']}))
    return results


def aggregator(state:BlogState):

    sorted_results = sorted(state['results'], key=lambda x: x['order'])
    sections_text  = "\n\n".join(r['content'] for r in sorted_results)

    final_blog = model.invoke(
        [
            SystemMessage(content=(
                "You are a senior editor and blog writer. "
                "You will receive a set of independently written blog sections and must merge them into one polished, cohesive blog post. "
                "Follow these rules:\n"
                "- Add a compelling H1 title at the top\n"
                "- Ensure smooth transitions between sections — rewrite transitions if they feel abrupt\n"
                "- Remove any repeated points or redundant sentences across sections\n"
                "- Keep the tone consistent throughout\n"
                "- Output clean markdown — headings, bullet points, bold text where appropriate\n"
                "- Do NOT add a generic closing like 'I hope you enjoyed this post'"
            )),
            HumanMessage(content=(
                f"Blog topic: {state['topic']}\n"
                f"Target audience: {state['plan'].target_audience}\n"
                f"Tone: {state['plan'].tone}\n"
                f"Tags: {', '.join(state['plan'].tags)}\n\n"
                "Here are the sections written by the team, in order:\n\n"
                f"{sections_text}\n\n"
                "Merge these into one final, publication-ready blog post in markdown."
            )),
        ]
    ).content

    file_name = f"final_blog_{state['topic'].replace(' ','_')}.md"
    with open(Path(__file__).parent / file_name, "w") as f:
        f.write(str(final_blog))

    return {"final_blog": final_blog}

#####################################################################
graph = StateGraph(BlogState)

graph.add_node("fanout", fanout)
graph.add_node("aggregator", aggregator)
graph.add_node("workerNode", workerNode)
graph.add_node("orchestorator", orchestorator)


graph.add_edge(START, "orchestorator")
graph.add_conditional_edges("orchestorator",fanout,["workerNode"])
graph.add_edge("workerNode", "aggregator")  
graph.add_edge("aggregator", END)


workflow = graph.compile()


response = workflow.invoke({"topic":"The future of AI in Politics"})


