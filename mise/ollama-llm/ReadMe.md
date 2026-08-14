# Ollama Quick Start Guide (macOS)

## What is Ollama?

**Ollama** is an open-source tool that allows you to run **Large Language Models (LLMs)** locally on your computer instead of using cloud services.

With Ollama you can:

- Run AI models completely offline
- Download and manage multiple LLMs
- Use models such as Llama, Mistral, Gemma, DeepSeek, Phi, and Qwen
- Expose a local REST API for your own applications
- Integrate with Python, LangChain, Open WebUI, n8n, and many other tools

By default, Ollama runs a local server at:

```
http://localhost:11434
```

---

# Install Ollama on macOS

## Option 1 (Recommended)

Using Homebrew:

```bash
brew install ollama
```

Start the Ollama service:

```bash
ollama serve
```

Keep this terminal running.

---

## Option 2

Download the installer from:

https://ollama.com/download

Install the application and launch it.

---

# Verify Installation

Check the version:

```bash
ollama --version
```

List installed models:

```bash
ollama list
```

---

# Download Your First Model
## you can see all the models in github https://ollama.com/search?c=cloud
Example using Llama 3.2:

```bash
ollama pull llama3.2
```

Run it:

```bash
ollama run llama3.2
```

Now you can chat directly from your terminal.

Exit with:

```text
/bye
```

or

```
Ctrl + D
```

---

# Useful Commands

## List installed models

```bash
ollama list
```

---

## Download a model

```bash
ollama pull mistral
```

Example:

```bash
ollama pull deepseek-r1
```

---

## Run a model

```bash
ollama run mistral
```

---

## Remove a model

```bash
ollama rm mistral
```

---

## Show model information

```bash
ollama show mistral
```

---

## List running models

```bash
ollama ps
```

---

## Stop a running model

```bash
ollama stop mistral
```

---

## Start the Ollama server

```bash
ollama serve
```

---

## View available commands

```bash
ollama --help
```

---

# Popular Models

| Model | Description |
|---------|-------------|
| llama3.2 | Meta's general-purpose model |
| mistral | Fast and lightweight |
| qwen3 | Excellent coding and reasoning |
| deepseek-r1 | Strong reasoning model |
| gemma3 | Google's open model |
| phi4 | Microsoft's compact model |
| codellama | Code generation |

---

# Using the REST API

Generate text:

```bash
curl http://localhost:11434/api/generate \
-d '{
  "model":"llama3.2",
  "prompt":"Explain Docker in simple words."
}'
```

List installed models:

```bash
curl http://localhost:11434/api/tags
```

---

# Example Workflow

1. Install Ollama
2. Start the server

```bash
ollama serve
```

3. Download a model

```bash
ollama pull llama3.2
```

4. Run the model

```bash
ollama run llama3.2
```

5. Use the REST API from your applications.

---

# Where Models Are Stored (macOS)

```
~/.ollama/models
```

---

# Common Tips

- Models are downloaded only once.
- The first run may take a few seconds while the model loads into memory.
- Larger models require more RAM.
- Ollama automatically starts a local API server on port **11434**.
- You can use multiple models on the same machine.

---

# Learn More

Official Website

https://ollama.com

Official Documentation

https://github.com/ollama/ollama

Model Library

https://ollama.com/library