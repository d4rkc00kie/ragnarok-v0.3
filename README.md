# With the same power and structure that could bring the end of days ! 
You all are warned , i did not named it R̅a̅g̅n̅a̅r̅o̅k̅ bc it sounds cool  , lulz 

Output without core value is only noise! 
R̅a̅g̅n̅a̅r̅o̅k̅ Strips away that noise. And makes it Clean, actionable insights in the terminal.
R̅a̅g̅n̅a̅r̅o̅k̅ is also trained to transform your OS into an weaponized AI controled system. and lives as an unity  3D avatar in your network or system !!! 
I removed most of the stuff that is unstable, and next version is not gonna be made public. 
The project ends here for the public so feel free to use it modify it and /or learn from it as you please. 

- **Markup-enhanced streaming** — answers flow as Markdown
- **Intelligent file handling** — open, read, analyze without breaking flow
- **Advanced command parsing** — *“open this folder and analyze the code”*
- **Real-time AI interaction** — always listening, always ready
- **Asynchronous folder analysis** — large trees off the UI thread
- **Interactive shell mode** — natural language → command → markup result
- **Contextual awareness** — files, shell, RAG only when relevant

## Run

```bash
ollama pull qwen3
ollama pull nomic-embed-text
cd ollama-tui && ./run.sh
```

## Natural language

| Example | Behavior |
|---------|----------|
| `open src and analyze the code` | Scan tree · structured insight |
| `read README.md` | Show file · actionable summary |
| `list .` | Directory listing |
| `run git status` / `git status` | Shell · markup output |
| `search auth middleware` | RAG + answer |

## Slash

`/model` `/params` `/tune` `/show` `/tools` `/rag` `/tts` `/avatar` `/clear` `/cd` `/export`

## Modules

| File | Role |
|------|------|
| `app.py` | Silent Precision TUI |
| `intent.py` | Natural-language intent |
| `workspace.py` | Async file/folder analysis |
| `shell_mode.py` | Safe shell + markup |
