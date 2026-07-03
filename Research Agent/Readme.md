# Blog Writer Agent

An AI-powered blog writing agent built with **LangGraph** and **LangChain** that takes a topic, creates a structured plan, gets human approval, then writes all sections in parallel and merges them into a final markdown blog post.

## How it works

```
Topic → Orchestrator (creates plan) → Human Review
                                            │
                              ┌─────────────┴─────────────┐
                         Approve                      Request Changes
                              │                            │
                    Workers run in parallel        Orchestrator re-plans
                    (one per section)                      │
                              │                     Human Review again
                    Aggregator merges                      
                              │
                    Final blog saved as .md
```

1. **Orchestrator** — calls the LLM to break the topic into 4–6 focused sections with word counts, keywords, tone, and target audience
2. **Human Review** — displays the plan in the terminal and waits for your approval or feedback
3. **Workers** — each section is written in parallel by a separate LLM call
4. **Aggregator** — merges all sections into one cohesive, publication-ready markdown file

## Setup

### 1. Clone and create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a file named `.env` in the **root of the project** (one level above this folder):

```
AI-Agents-Using-Langchain-and-LangGraph/
├── .env                  ← create this file here
├── requirements.txt
└── Research Agent/
    └── agent.py
```

Add the following keys to `.env`:

```env
# Required — OpenAI API key for GPT-4o-mini
OPENAI_API_KEY=sk-...your-key-here...
```

You can get your OpenAI API key from: https://platform.openai.com/api-keys

### 4. Start Redis

The agent uses Redis to save checkpoints so it can pause for human input and resume. Make sure Redis is running locally:

```bash
# Mac (with Homebrew)
brew install redis
brew services start redis

# Or run manually
redis-server
```

Redis runs on `localhost:6379` by default — no changes needed in the code.

### 5. Run the agent

```bash
cd "Research Agent"
python agent.py
```

## Human-in-the-loop

After the plan is generated you will see:

```
============================================================
         BLOG PLAN REVIEW
============================================================
Blog Title: ...
Description: ...
...
============================================================

What would you like to do?
  1. Accept plan — start writing
  2. Request changes — re-generate with your feedback
============================================================

Enter your choice (1 or 2):
```

- **Enter `1`** — accepts the plan and starts writing all sections in parallel
- **Enter `2`** — prompts you to describe what you want changed; the orchestrator re-plans using your feedback and the previous plan as context, then shows you the new plan for approval

## Output

The final blog is saved as a markdown file in the `Research Agent/` folder:

```
final_blog_<topic>.md
```

## Project structure

```
Research Agent/
├── agent.py          — main agent (orchestrator, workers, aggregator, human review)
├── Readme.md         — this file
└── final_blog_*.md   — generated blog outputs
```
