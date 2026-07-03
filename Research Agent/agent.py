from langgraph.graph import StateGraph,START,END
from typing import List, TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from langgraph.checkpoint.redis import RedisSaver
from pydantic import BaseModel,Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import operator
from langgraph.types import Command, Send,interrupt

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
    user_action_for_plan:Annotated[str,"User action for plan review"]
    user_feedbacks:Annotated[List[str],operator.add,"User feedback on the plan, if any"]
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

    feedbacks = state.get('user_feedbacks') or []
    feedback_text = ""
    if feedbacks and state.get('plan'):
        prev = state['plan']
        feedback_text = "\n\nThe user reviewed this previous plan and requested changes:\n"
        feedback_text += f"  Title: {prev.title}\n"
        feedback_text += f"  Sections: {', '.join(t.name for t in prev.tasks)}\n\n"
        feedback_text += "User feedback:\n"
        for fb in feedbacks:
            feedback_text += f"  - {fb}\n"
        feedback_text += "\nKeep what works, only change what the feedback asks for."

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
                f"that is informative, engaging, and optimized for search engines.{feedback_text}"
            )),
        ]
    )
    return {"plan": structured_llm_output}


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


def review_plan(state:BlogState):
    plan = state['plan']

    overall_plan = f"Blog Title: {plan.title}\n Description: {plan.description}\nTarget Audience: {plan.target_audience}\nTone: {plan.tone}\n\n\
        Below is the detailed plan for the blog:\n\n"
    for task in plan.tasks:
        overall_plan += (
            f"Section {task.order}: {task.name}\n"
            f"  Type: {task.section_type}\n"
            f"  Brief: {task.brief}\n"
            f"  Target word count: {task.word_count}\n"
            f"  SEO keywords: {', '.join(task.keywords) if task.keywords else 'none specified'}\n\n"
        )
    overall_plan += f"PLease review the above plan and confirm if it aligns with your expectations. If you have any changes or suggestions, please provide them now."
    user_action = interrupt({"type": "review_plan", "plan_summary": overall_plan.strip()})

    if user_action['action'] == "continue":
        return {"user_action_for_plan": user_action.get("action", ""),"plan": plan}
    elif user_action['action'] == "modify":
        return{"user_action_for_plan": user_action.get("action", ""),"user_feedbacks": [user_action.get("changes", "")]}

def check_user_action(state:BlogState):
    if state['user_action_for_plan'] == "continue":
        return [
            Send("workerNode", {"task": task, "topic": state['topic'], "plan": state['plan']})
            for task in state['plan'].tasks
        ]
    elif state['user_action_for_plan'] == "modify":
        return "orchestorator"
    else:
        raise ValueError("Invalid user action for plan review.")

#####################################################################
graph = StateGraph(BlogState)

graph.add_node("aggregator", aggregator)
graph.add_node("workerNode", workerNode)
graph.add_node("orchestorator", orchestorator)
graph.add_node("review_plan", review_plan)


graph.add_edge(START, "orchestorator")
graph.add_edge("orchestorator", "review_plan")
graph.add_conditional_edges("review_plan", check_user_action, ["workerNode", "orchestorator"])
graph.add_edge("workerNode", "aggregator")
graph.add_edge("aggregator", END)

###########################################################################################
CONN_URL = "redis://localhost:6379/0"

with RedisSaver.from_conn_string(CONN_URL) as checkpointer:
    checkpointer.setup()
    workflow = graph.compile(checkpointer=checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": "blog-session-42"}}

    response = workflow.invoke({"topic": "The future Job's in India in next 5 years"}, config=config)

    while "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0].value

        if interrupt_data["type"] == "review_plan":
            print("\n" + "="*60)
            print("         BLOG PLAN REVIEW")
            print("="*60)
            print(interrupt_data["plan_summary"])
            print("="*60)
            print("\nWhat would you like to do?")
            print("  1. Accept plan — start writing")
            print("  2. Request changes — re-generate with your feedback")
            print("="*60)

            while True:
                user_input = input("\nEnter your choice (1 or 2): ").strip()

                if user_input == "1":
                    print("\nPlan accepted. Starting blog generation...\n")
                    response = workflow.invoke(
                        Command(resume={"action": "continue"}), config=config
                    )
                    break

                elif user_input == "2":
                    print("\nDescribe the changes you want (be as specific as possible):")
                    changes = input("Your feedback: ").strip()
                    print("\nRe-generating plan with your feedback...\n")
                    response = workflow.invoke(
                        Command(resume={"action": "modify", "changes": changes}), config=config
                    )
                    break

                else:
                    print("Invalid choice. Please enter 1 or 2.")

