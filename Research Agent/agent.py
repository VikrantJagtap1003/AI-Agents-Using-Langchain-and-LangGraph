from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from langgraph.store.redis import RedisStore
from pydantic import BaseModel,Field
load_dotenv(Path(__file__).parent.parent / ".env")
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from langgraph.types import Send


#model 
model = ChatOpenAI(name="gpt-4o-mini", temperature=0.3)

#Input.  ---> AI call. --> output 

class BlogTask(BaseModel):
    id:str = Field(..., description="Unique ID of the task")
    name:str = Field(..., description="Name of the task")
    brief :str = Field(..., description="What things we need to cover in this task")

class Plan(BaseModel):
    title:str = Field(..., description="Title of the blog")
    description:str = Field(..., description="Description of the blog")
    Tasks: list[BlogTask] = Field(..., description="List of tasks for the blog")


class BlogState(TypedDict):
    topic: str
    plan: Plan
    results:Annotated[list[dict],operator.add]
    final_blog:Annotated[str, "Final blog content after all tasks are completed"]



#####################################################################


def workerNode(payload:dict) :

    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    result = model.invoke(
        [
            SystemMessage(content = "You are a blog writer. You will be given a task and you need to write the content for the task. "),
            HumanMessage(content = f"This task is part of a blog on the topic: {topic}. The task is: {task.name}. The brief for the task is: {task.brief}. Please write the content for this task."),
        ]
    ).content

    return {"results": [result]}


def orchestorator(state:BlogState):

    structured_llm_output = model.with_structured_output(Plan).invoke(
        [
            SystemMessage(content = "You are a blog planner. You will be given a topic and you need to create a plan for the blog."),
            HumanMessage(content = f"Create a plan for the blog on the topic: {state['topic']}"),
        ]
    )
    return {"plan": structured_llm_output}


def fanout(state:BlogState):
    tasks = state['plan'].Tasks
    results = []
    for task in tasks:
        results.append(Send("workerNode",{ "task":task ,"topic":state['topic'],"plan":state['plan']}))
    return results


def aggregator(state:BlogState):
    final_blog = model.invoke(
        [
            SystemMessage(content = "You are a blog writer. You will be given a topic and a plan for the blog. You need to write the final blog content based on the plan and the results of the tasks."),
            HumanMessage(content = f"Write the final blog content for the topic: {state['topic']} \n\n based on the plan: {state['plan']} \n\n and the results of the tasks: {state['results']}"),
        ]
    )
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

