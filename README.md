# Flowise to LangChain Converter

A CLI tool that converts [Flowise](https://flowiseai.com/) chatflow and agentflow JSON exports into executable LangChain TypeScript or Python code.

## Quick Start

```bash
git clone https://github.com/maximuslee1226/flowise-to-langchain.git
cd flowise-to-langchain
npm install && npm run build

# Convert a Flowise export to TypeScript
npx flowise-to-lc convert my-flow.json output/

# Convert with --overwrite if output dir exists
npx flowise-to-lc convert my-flow.json output/ --overwrite

# Convert to Python
npx flowise-to-lc convert my-flow.json output/ --format python
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `convert <input> <output>` | Convert Flowise JSON to LangChain code |
| `validate <input>` | Validate a Flowise JSON file without converting |
| `batch <glob> <output>` | Convert multiple files in parallel |
| `watch <input> <output>` | Watch a file and auto-convert on change |
| `run <input>` | Convert, install deps, and execute |
| `test <input>` | Run tests against converted code |

### Convert Options

```
--format <ts|js|python>   Output language (default: typescript)
--target <node|browser>   Runtime target (default: node)
--with-langfuse           Include Langfuse observability
--include-tests           Generate test files
--include-docs            Generate documentation
--overwrite               Overwrite existing output directory
```

## How It Works

The converter uses a four-stage pipeline:

```
Flowise JSON --> Parser --> IR (graph) --> Registry --> Emitter --> Output project
```

1. **Parser** (`src/parser/`) -- Zod-validated parsing of Flowise JSON with version detection (v1/v2)
2. **IR** (`src/ir/`) -- Builds an intermediate graph representation with topological sort and cycle detection
3. **Registry** (`src/registry/`) -- 145 registered node converters that map Flowise node types to LangChain code fragments
4. **Emitters** (`src/emitters/`) -- TypeScript or Python project generation from code fragments

## Supported Node Types (145 converters)

**LLM Providers**: OpenAI, ChatOpenAI, Anthropic, Azure OpenAI, Ollama, HuggingFace, Cohere, Replicate, Google, Bedrock

**Chains**: LLM, Conversation, Retrieval QA, Sequential, Transform, Map Reduce, Multi-Prompt

**Agents**: OpenAI Functions, Conversational, Tool Calling, Structured Chat, ReAct, Zero-Shot

**Memory**: Buffer, Buffer Window, Summary, Vector Store Retriever, Entity, Zep

**Tools**: Calculator, SerpAPI, Web Browser, Custom, HTTP, Python, Bash, Requests

**Vector Stores**: Pinecone, Chroma, FAISS, Weaviate, Qdrant, Supabase, Redis, Memory

**Embeddings**: OpenAI, HuggingFace, Cohere, Azure OpenAI, Google Vertex AI

**Document Loaders**: PDF, CSV, JSON, Text, Docx, Directory, Excel, Web

**Text Splitters**: Recursive Character, Character, Token, Markdown, LaTeX, HTML, Code

**AgentFlow V2**: Agent, Tool, Custom Function, Subflow, Start, Execute Flow, Human Input

**Search APIs**: Tavily, Brave, Google, Exa, Arxiv, WolframAlpha, SerpAPI, SearchAPI, DataForSEO, SearXNG, DuckDuckGo

**Business Tools**: Jira, Stripe, Airtable, Notion, Slack, HubSpot, Salesforce, Microsoft Teams, Asana

**Other**: Cache (Redis, InMemory, Momento, Upstash), Google Suite (8 tools), Output Parsers, RAG Chains, Streaming, Function Calling

## Project Structure

```
flowise-to-langchain/
  bin/                    CLI entry point
  src/
    cli/                  CLI commands (convert, validate, batch, watch, run, test)
    parser/               Flowise JSON parser with Zod schemas
    ir/                   Intermediate representation (graph, nodes, transformer)
    registry/             Converter registry + 30 converter modules
    emitters/
      typescript/         TypeScript project emitter
      python/             Python project emitter
    index.ts              Main FlowiseToLangChainConverter class
    converter.ts          ConverterPipeline with file I/O
  chatflows/              Sample Flowise JSON exports
  examples/               Example inputs (basic, complex, edge cases)
  test/                   Unit and integration tests
  docs/                   Documentation
```

## Example

Given a Flowise agentflow JSON with `startAgentflow -> executeFlowAgentflow -> humanInputAgentflow`, the converter generates:

**TypeScript** -- LangChain code with `StateGraph`, `axios` HTTP calls, `AzureChatOpenAI` summarization, and proper state management.

**Python** -- A standalone LangGraph script with `httpx` for API calls, `AzureChatOpenAI` for LLM summarization, `MemorySaver` checkpointing, and `interrupt_before` for human-in-the-loop.

## Development

```bash
npm run dev -- convert my-flow.json output/   # Run from source (no build needed)
npm run build                                  # Compile TypeScript
npm test                                       # Run tests
```

## License

MIT

## Acknowledgments

Originally created by Gregg Coppen ([@iaminawe](https://github.com/iaminawe)).
