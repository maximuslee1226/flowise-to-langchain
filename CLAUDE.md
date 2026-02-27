# Flowise to LangChain Converter

## What This Project Does

CLI tool that converts Flowise chatflow/agentflow JSON exports into LangChain TypeScript or Python code.

## Architecture

```
Flowise JSON --> Parser --> IR (graph) --> Registry --> Emitter --> Output code
```

- **Parser** (`src/parser/`): Zod-validated Flowise JSON parsing, version detection
- **IR** (`src/ir/`): Intermediate graph representation, topological sort, cycle detection
  - `nodes.ts`: Node type definitions, `NodeTemplates`, `StandardNodeFactory`
  - `transformer.ts`: `FlowiseToIRTransformer` + `IRToCodeTransformer` (code generation)
  - `graph.ts`: Graph analysis utilities
- **Registry** (`src/registry/`): 145 node converters that map Flowise types to LangChain code
  - `registry.ts`: `BaseConverter`, `ConverterRegistry`, `ConverterFactory`
  - `index.ts`: `BUILT_IN_CONVERTERS` array, `initializeRegistry()` with aliases
  - `converters/`: 30 converter modules (LLM, chain, agent, agentflow-v2, tools, etc.)
- **Emitters** (`src/emitters/`): Generate full project scaffolds
  - `typescript/`: TS project with package.json, .env.example, etc.
  - `python/`: Python project with requirements.txt, setup.py, etc.
- **CLI** (`src/cli/`): 6 commands -- convert, validate, batch, watch, run, test

## Key Code Paths

### Adding a new node converter

1. Add the Flowise type to `SupportedNodeType` union in `src/ir/nodes.ts`
2. Add a `NodeTemplate` entry in the `NodeTemplates` object (same file)
3. Create a converter class extending `BaseConverter` (or `BaseAgentFlowV2Converter` for agentflow types) in `src/registry/converters/`
4. Import and add it to `BUILT_IN_CONVERTERS` in `src/registry/index.ts`
5. Register aliases in `initializeRegistry()` (same file)
6. The `default` case in `IRToCodeTransformer.generateNodeCode()` (`src/ir/transformer.ts`) will automatically pick up registered converters via the registry

### How code generation dispatches

`IRToCodeTransformer.generateNodeCode()` in `src/ir/transformer.ts` has a switch statement for known types (openAI, chatOpenAI, llmChain, etc.). The `default` case checks the converter registry before falling back to a generic placeholder.

## Build & Test

```bash
npm run build          # Clean + compile + set permissions
npm run dev -- <cmd>   # Run from source via tsx
npm test               # Jest tests
```

The build uses `tsc -p tsconfig.build.json || true` (ignores type errors for incremental development).

## File Conventions

- All source is ESM (`"type": "module"` in package.json)
- Imports use `.js` extensions (TypeScript ESM convention)
- Converter classes use `readonly flowiseType` to declare the Flowise node name they handle
