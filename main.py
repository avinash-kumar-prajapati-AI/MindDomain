import os
import sys
from dotenv import load_dotenv

# Ensure the root project directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

from src import llm_client, graph_store, feedback_store, rag_pipeline

console = Console()

def run():
    # 1. Initialize databases
    console.print("[yellow]Initializing Database Schemas...[/]")
    try:
        graph_store.ensure_schema()
        feedback_store.init_db()
        console.print("[green]Database Schemas Initialized.[/]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize databases: {e}[/]")
        sys.exit(1)
        
    # 2. Check connection to LLM API
    console.print("[yellow]Verifying LLM Client connectivity...[/]")
    if not llm_client.health_check():
        console.print("[bold red]LLM connection check failed.[/]")
        console.print("Please verify that your NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY or BASE_URL are set correctly in your .env file.")
        sys.exit(1)
    console.print("[green]LLM Client successfully connected![/]")

    console.print(Panel.fit(
        "[bold cyan]Self-Improving Local GraphRAG Engine[/]\n"
        "Type a question to search. Type [bold red]'quit'[/] or [bold red]'exit'[/] to close.",
        title="Welcome", border_style="cyan"
    ))

    # Set up prompt history
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".prompt_history")
    session = PromptSession(history=FileHistory(history_file))
    
    state = "IDLE"
    last_response = None

    while True:
        try:
            if state == "IDLE":
                query = session.prompt("\n[You] > ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit"):
                    console.print("[cyan]Goodbye![/]")
                    break
                
                console.print("[dim]Retrieving and generating...[/]")
                last_response = rag_pipeline.answer(query)
                
                if last_response.needs_permission:
                    console.print(f"\n[bold green][RAG][/] {last_response.answer}")
                    state = "AWAITING_PERM"
                else:
                    console.print("\n[bold green][RAG][/]")
                    console.print(Markdown(last_response.answer))
                    
                    if last_response.citations:
                        console.print("\n[bold yellow]Citations:[/]")
                        for i, c in enumerate(last_response.citations, 1):
                            console.print(f"  [{i}] {c.title} — {c.url}")
                            
                    console.print("\n[dim]Was this helpful? [u]p / [d]own (or press Enter to skip)[/]")
                    state = "AWAITING_FB"

            elif state == "AWAITING_PERM":
                inp = session.prompt("[y/n] > ").strip().lower()
                if inp in ("y", "yes"):
                    console.print("[dim]Searching the web...[/]")
                    last_response = rag_pipeline.search_web_and_answer(last_response.raw_query)
                    console.print("\n[bold green][RAG][/]")
                    console.print(Markdown(last_response.answer))
                    
                    if last_response.citations:
                        console.print("\n[bold yellow]Citations:[/]")
                        for i, c in enumerate(last_response.citations, 1):
                            console.print(f"  [{i}] {c.title} — {c.url}")
                            
                    console.print("\n[dim]Was this helpful? [u]p / [d]own (or press Enter to skip)[/]")
                    state = "AWAITING_FB"
                else:
                    console.print("[dim]Skipping web search.[/]")
                    state = "IDLE"

            elif state == "AWAITING_FB":
                inp = session.prompt("[u/d] > ").strip().lower()
                if inp in ("u", "up", "d", "down"):
                    rating = "up" if inp in ("u", "up") else "down"
                    rag_pipeline.submit_feedback(last_response, rating)
                    label = "Saved to knowledge base." if rating == "up" else "Logged, not saved."
                    console.print(f"[dim green]{label}[/]")
                else:
                    console.print("[dim]Skipped feedback.[/]")
                state = "IDLE"

        except KeyboardInterrupt:
            console.print("\n[yellow]KeyboardInterrupt received. Returning to main prompt.[/]")
            state = "IDLE"
            continue
        except EOFError:
            console.print("\n[cyan]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred: {e}[/]")
            state = "IDLE"

if __name__ == "__main__":
    run()
