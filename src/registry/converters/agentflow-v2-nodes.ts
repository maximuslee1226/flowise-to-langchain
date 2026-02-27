/**
 * AgentFlow V2 Control-Flow Node Converters
 *
 * Converts Flowise AgentFlow V2 control-flow nodes:
 *   - startAgentflow      → Flow entry point / state initialisation
 *   - executeFlowAgentflow → HTTP call to an external Flowise chatflow
 *   - humanInputAgentflow  → LLM summary + human-in-the-loop interrupt
 */

import { BaseAgentFlowV2Converter } from './agentflow-v2.js';
import { IRNode, CodeFragment, GenerationContext } from '../../ir/types.js';

// ---------------------------------------------------------------------------
// StartAgentFlowConverter
// ---------------------------------------------------------------------------

export class StartAgentFlowConverter extends BaseAgentFlowV2Converter {
  readonly flowiseType = 'startAgentflow';

  protected getRequiredImports(): string[] {
    return ['StateGraph', 'END'];
  }

  protected getPackageName(): string {
    return '@langchain/langgraph';
  }

  protected getClassName(): string {
    return 'StateGraph';
  }

  protected getNodeType(): string {
    return 'agent';
  }

  protected extractNodeConfig(node: IRNode): Record<string, unknown> {
    return {
      inputType: this.getParameterValue(node, 'startInputType', 'chatInput'),
      ephemeralMemory: this.getParameterValue(node, 'startEphemeralMemory', false),
      flowState: this.getParameterValue(node, 'startState', []),
      persistState: this.getParameterValue(node, 'startPersistState', false),
    };
  }

  protected extractNodeState(node: IRNode): Record<string, unknown> {
    const flowState = this.getParameterValue<any[]>(node, 'startState', []);
    const state: Record<string, unknown> = { question: '', chat_history: [] };
    if (Array.isArray(flowState)) {
      for (const entry of flowState) {
        if (entry && typeof entry === 'object' && entry.key) {
          state[entry.key] = entry.value ?? '';
        }
      }
    }
    return state;
  }

  convert(node: IRNode, context: GenerationContext): CodeFragment[] {
    const variableName = this.generateVariableName(node, 'start_node');
    const config = this.extractNodeConfig(node);
    const stateEntries = this.extractNodeState(node);
    const fragments: CodeFragment[] = [];

    // Import
    fragments.push(
      this.createCodeFragment(
        `${node.id}_import`,
        'import',
        'import { StateGraph, END } from "@langchain/langgraph";\nimport { MemorySaver } from "@langchain/langgraph";',
        ['@langchain/langgraph'],
        node.id,
        1
      )
    );

    // State type
    const stateFields = Object.entries(stateEntries)
      .map(([key, value]) => {
        const tsType = Array.isArray(value) ? 'any[]' : typeof value === 'string' ? 'string' : 'any';
        return `  ${key}: ${tsType};`;
      })
      .join('\n');

    fragments.push(
      this.createCodeFragment(
        `${node.id}_state`,
        'initialization' as any,
        `// Flow state definition\ninterface FlowState {\n${stateFields}\n}`,
        ['state'],
        node.id,
        50
      )
    );

    // Entry-point node function
    const ephemeral = config.ephemeralMemory ? 'true' : 'false';
    const initCode = [
      `// Start node – validates input and initialises flow state`,
      `function ${variableName}(state: FlowState): FlowState {`,
      `  if (!state.question) throw new Error("A question is required to start the agent flow.");`,
      `  return {`,
      `    ...state,`,
      `    chat_history: ${ephemeral} ? [] : (state.chat_history ?? []),`,
      `  };`,
      `}`,
    ].join('\n');

    fragments.push(
      this.createCodeFragment(
        `${node.id}_init`,
        'initialization',
        initCode,
        ['StateGraph'],
        node.id,
        100
      )
    );

    // Post-init placeholder
    fragments.push(
      this.createCodeFragment(
        `${node.id}_post_init`,
        'initialization',
        `// ${variableName} post-initialization complete`,
        ['post-init'],
        node.id,
        125
      )
    );

    return fragments;
  }
}

// ---------------------------------------------------------------------------
// ExecuteFlowAgentFlowConverter
// ---------------------------------------------------------------------------

export class ExecuteFlowAgentFlowConverter extends BaseAgentFlowV2Converter {
  readonly flowiseType = 'executeFlowAgentflow';

  protected getRequiredImports(): string[] {
    return ['axios'];
  }

  protected getPackageName(): string {
    return 'axios';
  }

  protected getClassName(): string {
    return 'axios';
  }

  protected getNodeType(): string {
    return 'chain';
  }

  protected extractNodeConfig(node: IRNode): Record<string, unknown> {
    const rawInput = this.getParameterValue<string>(node, 'executeFlowInput', '{{ question }}') ?? '';
    // Strip Flowise HTML mention wrappers
    const inputExpr = rawInput.replace(/<[^>]+>/g, '').trim();

    return {
      selectedFlow: this.getParameterValue(node, 'executeFlowSelectedFlow', ''),
      input: inputExpr,
      baseURL: this.getParameterValue(node, 'executeFlowBaseURL', 'http://localhost:3000'),
      overrideConfig: this.getParameterValue(node, 'executeFlowOverrideConfig', ''),
      returnResponseAs: this.getParameterValue(node, 'executeFlowReturnResponseAs', 'userMessage'),
    };
  }

  protected extractNodeState(_node: IRNode): Record<string, unknown> {
    return { execute_flow_response: '' };
  }

  convert(node: IRNode, context: GenerationContext): CodeFragment[] {
    const variableName = this.generateVariableName(node, 'execute_flow_node');
    const config = this.extractNodeConfig(node);
    const fragments: CodeFragment[] = [];

    // Import
    fragments.push(
      this.createCodeFragment(
        `${node.id}_import`,
        'import',
        'import axios from "axios";',
        ['axios'],
        node.id,
        1
      )
    );

    // Node function
    const flowId = config.selectedFlow as string;
    const baseURL = config.baseURL as string;
    const inputExpr = config.input as string;
    const returnAs = config.returnResponseAs as string;

    const initCode = [
      `// Execute Flow node – calls external Flowise chatflow`,
      `async function ${variableName}(state: FlowState): Promise<Partial<FlowState>> {`,
      `  const url = \`${baseURL}/api/v1/prediction/${flowId}\`;`,
      `  const payload = { question: state.question };`,
      `  const headers: Record<string, string> = { "Content-Type": "application/json" };`,
      ``,
      `  const apiKey = process.env.FLOWISE_API_KEY;`,
      `  if (apiKey) headers["Authorization"] = \`Bearer \${apiKey}\`;`,
      ``,
      `  const { data } = await axios.post(url, payload, { headers, timeout: 120_000 });`,
      `  const answer: string = data.text ?? data.answer ?? JSON.stringify(data);`,
      ``,
      `  const history = [...(state.chat_history ?? [])];`,
      `  history.push({ role: "user", content: state.question });`,
      `  history.push({ role: "${returnAs === 'assistantMessage' ? 'assistant' : 'user'}", content: answer });`,
      ``,
      `  return { execute_flow_response: answer, chat_history: history };`,
      `}`,
    ].join('\n');

    fragments.push(
      this.createCodeFragment(
        `${node.id}_init`,
        'initialization',
        initCode,
        ['axios'],
        node.id,
        100
      )
    );

    // Post-init
    fragments.push(
      this.createCodeFragment(
        `${node.id}_post_init`,
        'initialization',
        `// ${variableName} post-initialization complete`,
        ['post-init'],
        node.id,
        125
      )
    );

    return fragments;
  }
}

// ---------------------------------------------------------------------------
// HumanInputAgentFlowConverter
// ---------------------------------------------------------------------------

export class HumanInputAgentFlowConverter extends BaseAgentFlowV2Converter {
  readonly flowiseType = 'humanInputAgentflow';

  protected getRequiredImports(): string[] {
    return ['AzureChatOpenAI'];
  }

  protected getPackageName(): string {
    return '@langchain/openai';
  }

  protected getClassName(): string {
    return 'AzureChatOpenAI';
  }

  protected getNodeType(): string {
    return 'agent';
  }

  protected extractNodeConfig(node: IRNode): Record<string, unknown> {
    const modelConfig = this.getParameterValue<Record<string, any>>(node, 'humanInputModelConfig', {}) ?? {};
    return {
      descriptionType: this.getParameterValue(node, 'humanInputDescriptionType', 'dynamic'),
      description: this.getParameterValue(node, 'humanInputDescription', ''),
      model: this.getParameterValue(node, 'humanInputModel', 'azureChatOpenAI'),
      prompt: this.getParameterValue(node, 'humanInputModelPrompt', ''),
      enableFeedback: this.getParameterValue(node, 'humanInputEnableFeedback', true),
      modelName: modelConfig.modelName ?? 'gpt-4o-mini',
      temperature: modelConfig.temperature ?? 0.9,
      streaming: modelConfig.streaming ?? true,
    };
  }

  protected extractNodeState(_node: IRNode): Record<string, unknown> {
    return { human_summary: '', human_feedback: null };
  }

  convert(node: IRNode, context: GenerationContext): CodeFragment[] {
    const variableName = this.generateVariableName(node, 'human_input_node');
    const config = this.extractNodeConfig(node);
    const fragments: CodeFragment[] = [];

    // Import
    fragments.push(
      this.createCodeFragment(
        `${node.id}_import`,
        'import',
        'import { AzureChatOpenAI } from "@langchain/openai";\nimport { HumanMessage, SystemMessage } from "@langchain/core/messages";',
        ['@langchain/openai', '@langchain/core/messages'],
        node.id,
        1
      )
    );

    // Strip HTML tags from the prompt for clean code output
    const rawPrompt = (config.prompt as string) || '';
    const cleanPrompt = rawPrompt
      .replace(/<[^>]+>/g, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
      .replace(/`/g, '\\`')
      .replace(/\$/g, '\\$');

    const modelName = config.modelName as string;
    const temperature = config.temperature as number;

    const initCode = [
      `// Human Input node – LLM-generated summary + human-in-the-loop interrupt`,
      `async function ${variableName}(state: FlowState): Promise<Partial<FlowState>> {`,
      `  const llm = new AzureChatOpenAI({`,
      `    model: "${modelName}",`,
      `    temperature: ${temperature},`,
      `    streaming: true,`,
      `  });`,
      ``,
      `  const systemPrompt = \`${cleanPrompt}\`;`,
      ``,
      `  const historyText = (state.chat_history ?? [])`,
      `    .map((m: any) => \`\${m.role}: \${m.content}\`)`,
      `    .join("\\n");`,
      ``,
      `  const response = await llm.invoke([`,
      `    new SystemMessage(systemPrompt),`,
      `    new HumanMessage(\`Conversation so far:\\n\${historyText}\`),`,
      `  ]);`,
      ``,
      `  return { human_summary: typeof response.content === "string" ? response.content : String(response.content) };`,
      `}`,
    ].join('\n');

    fragments.push(
      this.createCodeFragment(
        `${node.id}_init`,
        'initialization',
        initCode,
        ['AzureChatOpenAI'],
        node.id,
        100
      )
    );

    // Post-init
    fragments.push(
      this.createCodeFragment(
        `${node.id}_post_init`,
        'initialization',
        `// ${variableName} post-initialization complete`,
        ['post-init'],
        node.id,
        125
      )
    );

    return fragments;
  }
}
