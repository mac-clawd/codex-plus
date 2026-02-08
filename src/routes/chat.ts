/**
 * Chat completion routes - handles OpenAI and Claude API proxying
 */
import { Hono, Context } from 'hono'
import { stream } from 'hono/streaming'
import {
  createConverterState,
  processChunk,
  convertNonStreamingResponse,
} from '../utils/anthropic-to-openai-converter'
import {
  isCursorKeyCheck,
  createCursorBypassResponse,
  detectPotentialValidation,
  logPotentialValidation,
} from '../utils/cursor-byok-bypass'
import { logRequest, logResponse, logError, isVerbose, logHeaders, logStreamChunk } from '../utils/logger'
import { getChatGptInstructions } from '../utils/chatgpt-instructions'
import { convertToResponsesFormat } from '../utils/chat-to-responses'
import { decodeAccessToken } from '../oauth/crypto'
import { refreshClaudeToken } from '../auth/provider'
import { runCodexCli, type CodexCliSandbox } from '../utils/codex-cli-backend'
import {
  estimateRequestTokens,
  truncateMessages,
  CLAUDE_MAX_CONTEXT_TOKENS,
  SAFETY_MARGIN,
} from '../utils/token-estimation'

// Context overflow handling mode
export type ContextOverflowMode = 'truncate' | 'error' | 'warn'

// Module-level config (set via setContextOverflowMode)
let contextOverflowMode: ContextOverflowMode = 'truncate'

export function setContextOverflowMode(mode: ContextOverflowMode) {
  contextOverflowMode = mode
}

// Model alias mapping (short names → full Claude model IDs)
const MODEL_ALIASES: Record<string, string> = {
  'opus-4.5': 'claude-opus-4-5-20251101',
  'sonnet-4.5': 'claude-sonnet-4-5-20250514',
}

interface ModelMapping {
  from: string   // e.g., 'o3'
  to: string     // e.g., 'claude-opus-4-5-20251101'
}

interface KeyConfig {
  mappings: ModelMapping[]
  apiKey: string
  accountId?: string
}

interface ParsedKeys {
  configs: KeyConfig[]
  defaultKey?: string   // fallback for ultrathink/unmatched
  defaultAccountId?: string
  oauth?: OAuthTokens
  oauthError?: string
}

interface TokenInfo {
  token: string
  accountId?: string
}

interface OAuthTokens {
  claudeToken?: string
  claudeRefreshToken?: string
  chatgptToken?: string
  chatgptAccountId?: string
}

const CHATGPT_BASE_URL = process.env.CHATGPT_BASE_URL || 'https://chatgpt.com/backend-api/codex'
const CHATGPT_DEFAULT_MODEL = process.env.CHATGPT_DEFAULT_MODEL || 'gpt-5.2-codex'

// Backend selection for non-Claude requests:
// - auto (default): OpenAI key → OpenAI API, JWT/account_id → ChatGPT backend
// - codex-cli: spawn local `codex exec` and return an OpenAI-compatible response
const OPENAI_BACKEND = (process.env.OPENAI_BACKEND || 'auto').toLowerCase()

const CODEX_CLI_WORKDIR = process.env.CODEX_CLI_WORKDIR || process.cwd()
const CODEX_CLI_SANDBOX = (process.env.CODEX_CLI_SANDBOX || 'read-only') as CodexCliSandbox
const CODEX_CLI_TIMEOUT_MS = parseInt(process.env.CODEX_CLI_TIMEOUT_MS || '', 10) || 5 * 60_000
const CODEX_CLI_MODEL = process.env.CODEX_CLI_MODEL || ''

function splitProviderTokens(fullToken: string): string[] {
  if (!fullToken) return []
  const bySpace = fullToken.split(/\s+/).filter(Boolean)
  if (bySpace.length > 1) return bySpace

  const single = bySpace[0] || ''
  if (!single.includes(',')) return single ? [single] : []

  const lastColon = single.lastIndexOf(':')
  if (lastColon !== -1) {
    const mappingPart = single.slice(0, lastColon)
    const tokenPart = single.slice(lastColon + 1)
    if (tokenPart.includes(',')) {
      const splitTokens = tokenPart.split(',').map((t) => t.trim()).filter(Boolean)
      if (splitTokens.length > 0) {
        return [`${mappingPart}:${splitTokens[0]}`, ...splitTokens.slice(1)]
      }
    }
  }

  return single.split(',').map((t) => t.trim()).filter(Boolean)
}

function parseTokenWithAccount(token: string): TokenInfo {
  const hashIndex = token.indexOf('#')
  if (hashIndex > 0) {
    return {
      token: token.slice(0, hashIndex),
      accountId: token.slice(hashIndex + 1),
    }
  }
  return { token }
}

function isJwtToken(token: string): boolean {
  const parts = token.split('.')
  return parts.length === 3 && parts.every((p) => p.length > 0)
}

function mergeOAuthTokens(target: OAuthTokens, incoming: OAuthTokens | null) {
  if (!incoming) return
  if (incoming.claudeToken && !target.claudeToken) target.claudeToken = incoming.claudeToken
  if (incoming.claudeRefreshToken && !target.claudeRefreshToken) target.claudeRefreshToken = incoming.claudeRefreshToken
  if (incoming.chatgptToken && !target.chatgptToken) target.chatgptToken = incoming.chatgptToken
  if (incoming.chatgptAccountId && !target.chatgptAccountId) target.chatgptAccountId = incoming.chatgptAccountId
}

function parseOAuthToken(token: string): { tokens?: OAuthTokens; error?: string } | null {
  if (!token.startsWith('sb1.')) return null
  const payload = decodeAccessToken(token)
  if (!payload) {
    return { error: 'OAuth token invalid or expired. Please re-authenticate.' }
  }
  const result: OAuthTokens = {}
  if (payload.providers.claude?.access_token) result.claudeToken = payload.providers.claude.access_token
  if (payload.providers.claude?.refresh_token) result.claudeRefreshToken = payload.providers.claude.refresh_token
  if (payload.providers.chatgpt?.access_token) result.chatgptToken = payload.providers.chatgpt.access_token
  if (payload.providers.chatgpt?.account_id) result.chatgptAccountId = payload.providers.chatgpt.account_id
  return { tokens: result }
}

function isSubBridgeToken(token: string): boolean {
  return token.startsWith('sb1.')
}

function isCodexCliSentinelKey(token: string | undefined): boolean {
  if (!token) return false
  const t = token.trim().toLowerCase()
  return t === 'codex' || t === 'codex-cli' || t === 'codex-plus' || t === 'codex+'
}

function normalizeChatGptModel(requestedModel: string): string {
  if (!requestedModel) return CHATGPT_DEFAULT_MODEL
  if (requestedModel.includes('codex')) return requestedModel
  if (requestedModel.startsWith('gpt-5.2')) return CHATGPT_DEFAULT_MODEL
  if (requestedModel.startsWith('gpt-5')) return CHATGPT_DEFAULT_MODEL
  return CHATGPT_DEFAULT_MODEL
}

function messageContentToText(content: any): string {
  if (content == null) return ''
  if (typeof content === 'string') return content

  if (Array.isArray(content)) {
    const parts: string[] = []
    for (const block of content) {
      if (block == null) continue
      if (typeof block === 'string') {
        parts.push(block)
        continue
      }
      if (typeof block !== 'object') {
        parts.push(String(block))
        continue
      }

      const type = (block as any).type
      if ((type === 'text' || type === 'input_text' || type === 'output_text') && typeof (block as any).text === 'string') {
        parts.push((block as any).text)
        continue
      }

      if (type === 'image_url') {
        const url = typeof (block as any).image_url === 'string'
          ? (block as any).image_url
          : (block as any).image_url?.url
        parts.push(url ? `[image: ${url}]` : '[image]')
        continue
      }

      // Best-effort fallback for unknown blocks.
      try {
        parts.push(JSON.stringify(block))
      } catch {
        parts.push(String(block))
      }
    }
    return parts.join('\n')
  }

  try {
    return JSON.stringify(content)
  } catch {
    return String(content)
  }
}

function buildCodexCliPrompt(body: any): string {
  let messages: any[] = Array.isArray(body.messages) ? body.messages : []
  if (messages.length === 0 && body.input !== undefined) {
    if (typeof body.input === 'string') messages = [{ role: 'user', content: body.input }]
    else if (Array.isArray(body.input)) messages = body.input
  }

  const lines: string[] = []

  // Preserve system messages up-front; Codex CLI will still apply its own internal system prompt.
  const systemMessages = messages.filter((m) => m?.role === 'system')
  if (systemMessages.length > 0) {
    lines.push('System:')
    for (const m of systemMessages) {
      const text = messageContentToText(m?.content)
      if (text.trim()) lines.push(text.trimEnd())
    }
    lines.push('')
  }

  // Render the rest of the conversation with explicit role tags.
  const convo = messages.filter((m) => m?.role !== 'system')
  for (const m of convo) {
    const role = String(m?.role || 'user').toUpperCase()

    // Tool/function call metadata (Cursor / OpenAI chat format)
    if (role === 'ASSISTANT' && Array.isArray(m?.tool_calls) && m.tool_calls.length > 0) {
      lines.push('ASSISTANT (tool_calls):')
      for (const tc of m.tool_calls) {
        const name = tc?.function?.name || tc?.name || 'unknown'
        const args = tc?.function?.arguments || tc?.arguments || ''
        lines.push(`${name}(${typeof args === 'string' ? args : JSON.stringify(args)})`)
      }
      lines.push('')
      continue
    }

    const text = messageContentToText(m?.content)
    if (!text.trim()) continue
    lines.push(`${role}:`)
    lines.push(text.trimEnd())
    lines.push('')
  }

  // Nudge Codex CLI to respond as the assistant.
  lines.push('ASSISTANT:')
  return lines.join('\n')
}

/**
 * Parse routed API keys from Authorization header
 * Format: "o3=opus-4.5,o3-mini=sonnet-4.5:sk-ant-xxx sk-xxx" (space-separated tokens; comma fallback supported) or just "sk-ant-xxx" for default
 */
function parseRoutedKeys(authHeader: string | undefined): ParsedKeys {
  if (!authHeader) return { configs: [] }
  const fullToken = authHeader.replace(/^Bearer\s+/i, '').trim()

  // Split by space (or comma fallback) to handle multiple provider tokens
  const tokens = splitProviderTokens(fullToken)
  const configs: KeyConfig[] = []
  let defaultKey: string | undefined
  let defaultAccountId: string | undefined
  const oauthTokens: OAuthTokens = {}
  let oauthError: string | undefined

  for (const token of tokens) {
    if (!token) continue

    // Check if it's a routed key (contains '=')
    if (!token.includes('=')) {
      const parsed = parseTokenWithAccount(token)
      if (isSubBridgeToken(parsed.token)) {
        const oauthParsed = parseOAuthToken(parsed.token)
        if (oauthParsed?.error) oauthError = oauthParsed.error
        if (oauthParsed?.tokens) mergeOAuthTokens(oauthTokens, oauthParsed.tokens)
        continue
      }
      // Plain key without routing - use as default
      if (!defaultKey) {
        defaultKey = parsed.token
        defaultAccountId = parsed.accountId
      }
      continue
    }

    // Split by last colon to separate mappings from key
    const lastColon = token.lastIndexOf(':')
    if (lastColon === -1) {
      // No colon found in routed key, skip it
      continue
    }

    const mappingsPart = token.slice(0, lastColon)
    const parsedToken = parseTokenWithAccount(token.slice(lastColon + 1))
    let apiKey = parsedToken.token
    if (isSubBridgeToken(apiKey)) {
      const oauthParsed = parseOAuthToken(apiKey)
      if (oauthParsed?.error) {
        oauthError = oauthParsed.error
        continue
      }
      if (oauthParsed?.tokens) {
        mergeOAuthTokens(oauthTokens, oauthParsed.tokens)
        if (oauthParsed.tokens.claudeToken) {
          apiKey = oauthParsed.tokens.claudeToken
        } else {
          continue
        }
      }
    }

    const mappings = mappingsPart.split(',').map(m => {
      const [from, to] = m.split('=')
      const resolvedTo = MODEL_ALIASES[to] || to
      return { from: from.trim(), to: resolvedTo }
    })

    configs.push({ mappings, apiKey, accountId: parsedToken.accountId })
  }

  const hasOAuth = Boolean(oauthTokens.claudeToken || oauthTokens.chatgptToken || oauthTokens.chatgptAccountId)
  if (hasOAuth || oauthError) {
    return { configs, defaultKey, defaultAccountId, oauth: hasOAuth ? oauthTokens : undefined, oauthError }
  }
  return { configs, defaultKey, defaultAccountId }
}

/**
 * Check if a model name resolves to a Claude model (either directly or via alias)
 */
function isClaudeModel(model: string): boolean {
  return model.startsWith('claude-') || MODEL_ALIASES[model]?.startsWith('claude-')
}

/**
 * Find the Claude model and API key for a given requested model
 */
function resolveModelRouting(requestedModel: string, parsedKeys: ParsedKeys): { claudeModel: string; apiKey: string } | null {
  // Check all configs for a matching route
  for (const config of parsedKeys.configs) {
    for (const mapping of config.mappings) {
      if (mapping.from === requestedModel) {
        return { claudeModel: mapping.to, apiKey: config.apiKey }
      }
    }
  }

  // Resolve model alias (e.g., 'opus-4.5' -> 'claude-opus-4-5-20251101')
  const resolvedModel = MODEL_ALIASES[requestedModel] || requestedModel

  // If model is a Claude model (directly or via alias), use default key
  if (isClaudeModel(requestedModel) && parsedKeys.defaultKey) {
    return { claudeModel: resolvedModel, apiKey: parsedKeys.defaultKey }
  }

  // Fallback to default key with the model as-is (for ultrathink)
  if (parsedKeys.defaultKey) {
    return { claudeModel: resolvedModel, apiKey: parsedKeys.defaultKey }
  }

  return null
}

// Convert OpenAI content block to Claude format (handles image_url -> image conversion)
function convertContentBlockToClaude(block: any): any {
  if (!block || typeof block !== 'object') return block

  // Text blocks pass through unchanged
  if (block.type === 'text') {
    return { type: 'text', text: block.text || '' }
  }

  // Convert OpenAI image_url to Claude image format
  if (block.type === 'image_url') {
    const url = typeof block.image_url === 'string'
      ? block.image_url
      : block.image_url?.url

    if (!url) return null

    // Handle data URLs (base64 encoded images)
    if (url.startsWith('data:')) {
      const match = url.match(/^data:([^;]+);base64,(.+)$/)
      if (match) {
        const [, mediaType, data] = match
        return {
          type: 'image',
          source: {
            type: 'base64',
            media_type: mediaType,
            data: data,
          },
        }
      }
    }

    // Handle external URLs
    return {
      type: 'image',
      source: {
        type: 'url',
        url: url,
      },
    }
  }

  // Pass through other block types unchanged
  return block
}

function convertMessages(messages: any[]): any[] {
  const converted: any[] = []

  for (const msg of messages) {
    if (msg.type === 'custom_tool_call' || msg.type === 'function_call') {
      let toolInput = msg.input || msg.arguments
      if (typeof toolInput === 'string') {
        try { toolInput = JSON.parse(toolInput) } catch { toolInput = { command: toolInput } }
      }
      const toolUse = { type: 'tool_use', id: msg.call_id, name: msg.name, input: toolInput || {} }
      const last = converted[converted.length - 1]
      if (last?.role === 'assistant' && Array.isArray(last.content)) last.content.push(toolUse)
      else converted.push({ role: 'assistant', content: [toolUse] })
      continue
    }

    if (msg.type === 'custom_tool_call_output' || msg.type === 'function_call_output') {
      const toolResult = { type: 'tool_result', tool_use_id: msg.call_id, content: msg.output || '' }
      const last = converted[converted.length - 1]
      if (last?.role === 'user' && Array.isArray(last.content) && last.content[0]?.type === 'tool_result') last.content.push(toolResult)
      else converted.push({ role: 'user', content: [toolResult] })
      continue
    }

    if (!msg.role) continue

    if (msg.role === 'assistant' && msg.tool_calls?.length) {
      let content: any[] = []
      if (msg.content) {
        if (typeof msg.content === 'string') {
          content = [{ type: 'text', text: msg.content }]
        } else if (Array.isArray(msg.content)) {
          // Preserve existing content blocks, converting to Claude format
          for (const block of msg.content) {
            if (block.type === 'text' && typeof block.text === 'string') {
              content.push({ type: 'text', text: block.text })
            } else if (block.type === 'tool_use') {
              content.push(block) // Already Claude format
            }
          }
        }
      }
      for (const tc of msg.tool_calls) {
        let input = tc.function?.arguments || tc.arguments || {}
        if (typeof input === 'string') {
          try {
            input = JSON.parse(input || '{}')
          } catch {
            input = { raw: input }
          }
        }
        content.push({
          type: 'tool_use', id: tc.id, name: tc.function?.name || tc.name, input
        })
      }
      converted.push({ role: 'assistant', content })
      continue
    }

    if (msg.role === 'tool') {
      const toolResult = { type: 'tool_result', tool_use_id: msg.tool_call_id, content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content) }
      const last = converted[converted.length - 1]
      if (last?.role === 'user' && Array.isArray(last.content) && last.content[0]?.type === 'tool_result') last.content.push(toolResult)
      else converted.push({ role: 'user', content: [toolResult] })
      continue
    }

    // Handle assistant messages with array content (no tool_calls)
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      const content: any[] = []
      for (const block of msg.content) {
        if (block.type === 'text' && typeof block.text === 'string') {
          content.push({ type: 'text', text: block.text })
        } else if (block.type === 'tool_use') {
          content.push(block)
        }
      }
      converted.push({ role: 'assistant', content: content.length > 0 ? content : '' })
      continue
    }

    // Handle user messages with array content (may contain image_url blocks)
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      const content: any[] = []
      for (const block of msg.content) {
        const convertedBlock = convertContentBlockToClaude(block)
        if (convertedBlock) content.push(convertedBlock)
      }
      converted.push({ role: 'user', content: content.length > 0 ? content : '' })
      continue
    }

    converted.push({ role: msg.role, content: msg.content ?? '' })
  }

  // Ensure all assistant messages have non-empty content (required by Anthropic API:
  // "all messages must have non-empty content except for the optional final assistant message")
  for (const msg of converted) {
    if (msg.role !== 'assistant') continue

    if (typeof msg.content === 'string') {
      msg.content = msg.content.trimEnd() || '...'
    } else if (Array.isArray(msg.content)) {
      if (msg.content.length === 0) {
        msg.content = [{ type: 'text', text: '...' }]
      } else {
        for (const block of msg.content) {
          if (block.type === 'text') {
            block.text = (block.text?.trimEnd()) || '...'
          }
        }
      }
    }
  }

  return converted
}

function buildChatGptResponsesBody(body: any, requestedModel: string, isStreaming: boolean) {
  // ChatGPT backend requires EXACTLY the base codex prompt as instructions
  const instructions = getChatGptInstructions()

  // Use the robust converter to handle all message formats
  const { input, developerMessages } = convertToResponsesFormat(body)

  // Prepend developer messages to input
  const fullInput = [...developerMessages, ...input]

  // Fallback if no input
  if (fullInput.length === 0) {
    fullInput.push({
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text: 'Hello.' }],
    })
  }

  const tools = Array.isArray(body.tools) ? body.tools : []

  return {
    model: requestedModel,
    instructions,
    input: fullInput,
    tools,
    tool_choice: 'auto',
    parallel_tool_calls: false,
    stream: true,  // Backend REQUIRES stream: true
    store: false,
  }
}

function createChatGptStreamState(model: string) {
  const now = Math.floor(Date.now() / 1000)
  return {
    buffer: '',
    id: `chatcmpl-${Date.now().toString(36)}`,
    model,
    created: now,
    roleSent: false,
    sawTextDelta: false,
    toolCallsSeen: false,
    toolCallIndex: 0,
    processedItemIds: new Set<string>(),  // Track processed item IDs to prevent duplicates
  }
}

function createChatChunk(state: ReturnType<typeof createChatGptStreamState>, delta: any, finishReason: string | null, usage?: any) {
  return {
    id: state.id,
    object: 'chat.completion.chunk',
    created: state.created,
    model: state.model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
      },
    ],
    ...(usage ? { usage } : {}),
  }
}

function mapUsage(usage: any) {
  if (!usage) return undefined
  const prompt = usage.input_tokens ?? usage.prompt_tokens ?? 0
  const completion = usage.output_tokens ?? usage.completion_tokens ?? 0
  const total = usage.total_tokens ?? (prompt + completion)
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
  }
}

function processChatGptChunk(state: ReturnType<typeof createChatGptStreamState>, chunk: string) {
  state.buffer += chunk
  const results: Array<{ type: 'chunk' | 'done'; data?: any }> = []
  const parts = state.buffer.split('\n\n')
  state.buffer = parts.pop() || ''

  for (const part of parts) {
    const lines = part.split('\n')
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (!data || data === '[DONE]') continue
      let payload: any
      try {
        payload = JSON.parse(data)
      } catch {
        continue
      }
      const kind = payload?.type
      if (kind === 'response.created' && payload.response?.id) {
        state.id = `chatcmpl-${String(payload.response.id).replace(/^resp_/, '')}`
        continue
      }
      if (kind === 'response.output_text.delta' && typeof payload.delta === 'string') {
        const delta: any = { content: payload.delta }
        if (!state.roleSent) {
          delta.role = 'assistant'
          state.roleSent = true
        }
        state.sawTextDelta = true
        results.push({ type: 'chunk', data: createChatChunk(state, delta, null) })
        continue
      }
      if (kind === 'response.output_item.done' && payload.item) {
        const item = payload.item

        // Generate item identifier and skip if already processed
        const itemId = item.id || item.call_id || `${item.type}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
        if (state.processedItemIds.has(itemId)) {
          continue  // Already processed this item, skip to prevent duplicates
        }
        state.processedItemIds.add(itemId)

        if (item.type === 'message' && item.role === 'assistant' && !state.sawTextDelta) {
          const blocks = Array.isArray(item.content) ? item.content : []
          const text = blocks
            .filter((b: any) => b?.type === 'output_text' && typeof b.text === 'string')
            .map((b: any) => b.text)
            .join('')
          if (text) {
            const delta: any = { content: text }
            if (!state.roleSent) {
              delta.role = 'assistant'
              state.roleSent = true
            }
            results.push({ type: 'chunk', data: createChatChunk(state, delta, null) })
          }
        } else if (item.type === 'function_call') {
          state.toolCallsSeen = true
          const toolCallId = item.call_id || item.id || `call_${state.toolCallIndex}`

          const delta: any = {
            tool_calls: [
              {
                index: state.toolCallIndex++,
                id: toolCallId,
                type: 'function',
                function: {
                  name: item.name || 'unknown',
                  arguments: item.arguments || '',
                },
              },
            ],
          }
          if (!state.roleSent) {
            delta.role = 'assistant'
            state.roleSent = true
          }
          results.push({ type: 'chunk', data: createChatChunk(state, delta, null) })
        }
        continue
      }
      if (kind === 'response.completed') {
        const usage = mapUsage(payload.response?.usage)
        const finish = state.toolCallsSeen ? 'tool_calls' : 'stop'
        results.push({ type: 'chunk', data: createChatChunk(state, {}, finish, usage) })
        results.push({ type: 'done' })
      }
    }
  }

  return results
}

async function handleOpenAIProxy(c: Context, body: any, requestedModel: string, openaiToken: string, isStreaming: boolean) {
  logRequest('openai', requestedModel, {})

  // Forward request directly to OpenAI API
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'authorization': `Bearer ${openaiToken}`,
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorText = await response.text()
    logError(errorText.slice(0, 200))

    // Try to parse OpenAI error
    try {
      const openAIError = JSON.parse(errorText)
      return c.json(openAIError, response.status as any)
    } catch (parseError) {
      return new Response(errorText, { status: response.status })
    }
  }

  logResponse(response.status)

  // For streaming, pass through the stream
  if (isStreaming) {
    return new Response(response.body, {
      status: response.status,
      headers: {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        'connection': 'keep-alive',
      },
    })
  } else {
    // For non-streaming, pass through the JSON response
    const responseData = await response.json()
    return c.json(responseData)
  }
}

async function handleChatGptProxy(
  c: Context,
  body: any,
  requestedModel: string,
  tokenInfo: TokenInfo,
  isStreaming: boolean,
) {
  const chatgptModel = normalizeChatGptModel(requestedModel)
  const responseBody = buildChatGptResponsesBody(body, chatgptModel, isStreaming)
  logRequest('chatgpt', `${requestedModel} → ${chatgptModel}`, {
    system: responseBody.instructions,
    messages: responseBody.input,
    tools: responseBody.tools,
  })

  if (!tokenInfo.accountId) {
    return c.json({
      error: {
        message: 'ChatGPT account id missing. Re-login to refresh your ChatGPT token.',
        type: 'authentication_error',
        code: 'authentication_error',
      }
    }, 401)
  }

  const baseUrl = CHATGPT_BASE_URL.replace(/\/$/, '')

  if (isVerbose()) {
    logHeaders('Request Headers', {
      'content-type': 'application/json',
      'authorization': `Bearer ${tokenInfo.token}`,
      'chatgpt-account-id': tokenInfo.accountId || '',
      'originator': 'codex_cli_rs',
      'accept': 'text/event-stream',
    })
  }

  const response = await fetch(`${baseUrl}/responses`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'authorization': `Bearer ${tokenInfo.token}`,
      'chatgpt-account-id': tokenInfo.accountId,
      'originator': 'codex_cli_rs',
      'accept': 'text/event-stream',  // Backend always requires streaming
    },
    body: JSON.stringify(responseBody),
  })

  if (!response.ok) {
    const errorText = await response.text()
    logError(`ChatGPT API error (${response.status}): ${errorText.slice(0, 500)}`)
    try {
      const parsed = JSON.parse(errorText)
      // ChatGPT can return errors in various formats
      const errorMessage = parsed.error?.message || parsed.message || parsed.detail || errorText.slice(0, 200) || 'Unknown error'
      const errorType = parsed.error?.type || parsed.type || 'api_error'

      // Map error types to user-friendly messages
      let userMessage = errorMessage
      if (response.status === 401 || errorType === 'authentication_error') {
        userMessage = `ChatGPT authentication failed: ${errorMessage}. Try re-logging in.`
      } else if (response.status === 429 || errorType === 'rate_limit_error') {
        userMessage = `ChatGPT rate limit exceeded: ${errorMessage}`
      } else if (response.status === 400 || errorType === 'invalid_request_error') {
        userMessage = `Invalid request to ChatGPT: ${errorMessage}`
      }

      return c.json({
        error: {
          message: userMessage,
          type: errorType,
          code: errorType,
        }
      }, response.status as any)
    } catch {
      return c.json({
        error: {
          message: `ChatGPT error (${response.status}): ${errorText.slice(0, 300)}`,
          type: 'api_error',
          code: 'api_error',
        }
      }, response.status as any)
    }
  }

  logResponse(response.status)

  if (isVerbose()) {
    const respHeaders: Record<string, string> = {}
    response.headers.forEach((value, key) => {
      respHeaders[key] = value
    })
    logHeaders('Response Headers', respHeaders)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  const state = createChatGptStreamState(chatgptModel)

  if (isStreaming) {
    return stream(c, async (s) => {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        if (isVerbose()) {
          logStreamChunk(chunk)
        }
        const results = processChatGptChunk(state, chunk)
        for (const result of results) {
          if (result.type === 'chunk') {
            await s.write(`data: ${JSON.stringify(result.data)}\n\n`)
          } else if (result.type === 'done') {
            await s.write('data: [DONE]\n\n')
          }
        }
      }
      reader.releaseLock()
    })
  }

  // Non-streaming: aggregate the stream into a final response
  let fullContent = ''
  const toolCallsMap = new Map<string, any>()  // Use Map for deduplication by ID
  let usage: any = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value, { stream: true })
    if (isVerbose()) {
      logStreamChunk(chunk)
    }
    const results = processChatGptChunk(state, chunk)
    for (const result of results) {
      if (result.type === 'chunk' && result.data?.choices?.[0]?.delta) {
        const delta = result.data.choices[0].delta
        if (delta.content) fullContent += delta.content

        // Aggregate tool calls - merge by index, deduplicate by ID
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (tc.id) {
              // New tool call with ID - store by ID
              if (!toolCallsMap.has(tc.id)) {
                toolCallsMap.set(tc.id, { ...tc })
              }
            } else if (tc.index !== undefined) {
              // Continuation chunk (has index but no id) - find and merge arguments
              // Find existing tool call by index
              for (const [id, existing] of toolCallsMap) {
                if (existing.index === tc.index && tc.function?.arguments) {
                  existing.function = existing.function || {}
                  existing.function.arguments = (existing.function.arguments || '') + tc.function.arguments
                  break
                }
              }
            }
          }
        }

        if (result.data.usage) usage = result.data.usage
      }
    }
  }
  reader.releaseLock()

  // Convert Map to array
  const toolCalls = Array.from(toolCallsMap.values())

  return c.json({
    id: state.id,
    object: 'chat.completion',
    created: state.created,
    model: chatgptModel,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: fullContent || null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: toolCalls.length > 0 ? 'tool_calls' : 'stop',
    }],
    usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  })
}

async function handleCodexCliProxy(
  c: Context,
  body: any,
  requestedModel: string,
  isStreaming: boolean,
) {
  const codexModel = CODEX_CLI_MODEL || (requestedModel.includes('codex') ? requestedModel : CHATGPT_DEFAULT_MODEL)
  const prompt = buildCodexCliPrompt(body)

  logRequest('codex-cli', `${requestedModel} → ${codexModel}`, {
    cwd: CODEX_CLI_WORKDIR,
    sandbox: CODEX_CLI_SANDBOX,
    promptPreview: prompt.slice(0, 4000),
  })

  let resultText = ''
  let usage: any = null
  try {
    const res = await runCodexCli(prompt, {
      cwd: CODEX_CLI_WORKDIR,
      sandbox: CODEX_CLI_SANDBOX,
      timeoutMs: CODEX_CLI_TIMEOUT_MS,
      model: CODEX_CLI_MODEL || (requestedModel.includes('codex') ? requestedModel : undefined),
    })
    resultText = res.text || ''
    usage = res.usage || null
  } catch (err: any) {
    const msg = typeof err?.message === 'string' ? err.message : String(err)
    logError(`Codex CLI error: ${msg.slice(0, 500)}`)
    return c.json({
      error: {
        message: `Codex CLI backend failed: ${msg}`,
        type: 'api_error',
        code: 'codex_cli_error',
      }
    }, 500)
  }

  logResponse(200)

  const created = Math.floor(Date.now() / 1000)
  const id = `chatcmpl-${Date.now().toString(36)}`

  const promptTokens = usage?.input_tokens ?? 0
  const completionTokens = usage?.output_tokens ?? 0
  const totalTokens = (usage?.total_tokens ?? (promptTokens + completionTokens)) || 0
  const mappedUsage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  }

  if (isStreaming) {
    return stream(c, async (s) => {
      // 1) role chunk
      await s.write(`data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model: codexModel,
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      })}\n\n`)

      // 2) content chunk (single shot; Codex CLI doesn't stream deltas)
      await s.write(`data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model: codexModel,
        choices: [{ index: 0, delta: { content: resultText }, finish_reason: null }],
      })}\n\n`)

      // 3) final chunk with usage
      await s.write(`data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model: codexModel,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        usage: mappedUsage,
      })}\n\n`)

      await s.write('data: [DONE]\n\n')
    })
  }

  return c.json({
    id,
    object: 'chat.completion',
    created,
    model: codexModel,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: resultText || null,
      },
      finish_reason: 'stop',
    }],
    usage: mappedUsage,
  })
}

async function handleChatCompletion(c: Context) {
  const body = await c.req.json()
  const requestedModel = body.model || ''
  const isStreaming = body.stream === true

  if (isCursorKeyCheck(body)) {
    logRequest('bypass', requestedModel, {})
    logResponse(200)
    return c.json(createCursorBypassResponse(requestedModel))
  }

  // In verbose mode, log requests that look like potential validation attempts
  // This helps identify new Cursor validation patterns for future bypass improvements
  const validationReason = detectPotentialValidation(body)
  if (validationReason) {
    logPotentialValidation(body, validationReason)
  }

  const parsedKeys = parseRoutedKeys(c.req.header('authorization'))
  if (parsedKeys.oauthError) {
    return c.json({
      error: {
        message: parsedKeys.oauthError,
        type: 'authentication_error',
        code: 'authentication_error',
      }
    }, 401)
  }
  const oauthTokens = parsedKeys.oauth
  let routing = resolveModelRouting(requestedModel, parsedKeys)
  if (!routing && oauthTokens?.claudeToken) {
    const resolvedModel = MODEL_ALIASES[requestedModel] || requestedModel
    if (resolvedModel.startsWith('claude-')) {
      routing = { claudeModel: resolvedModel, apiKey: oauthTokens.claudeToken }
    }
  }
  const isClaude = routing !== null && (
    routing.claudeModel.startsWith('claude-') ||
    isClaudeModel(requestedModel) ||
    parsedKeys.configs.some(c => c.mappings.some(m => m.from === requestedModel))
  )

  // If not a Claude model and we have a default key, proxy to OpenAI or ChatGPT backend
  if (!isClaude) {
    const useCodexCli = OPENAI_BACKEND === 'codex-cli' || isCodexCliSentinelKey(parsedKeys.defaultKey)
    if (useCodexCli) {
      return handleCodexCliProxy(c, body, requestedModel, isStreaming)
    }

    if (parsedKeys.defaultKey) {
      const tokenInfo: TokenInfo = {
        token: parsedKeys.defaultKey,
        accountId: parsedKeys.defaultAccountId,
      }
      const useChatGpt = Boolean(tokenInfo.accountId) || isJwtToken(tokenInfo.token)
      if (useChatGpt) {
        return handleChatGptProxy(c, body, requestedModel, tokenInfo, isStreaming)
      }
      return handleOpenAIProxy(c, body, requestedModel, tokenInfo.token, isStreaming)
    }
    if (oauthTokens?.chatgptToken) {
      return handleChatGptProxy(c, body, requestedModel, {
        token: oauthTokens.chatgptToken,
        accountId: oauthTokens.chatgptAccountId,
      }, isStreaming)
    }
  }

  if (!isClaude || !routing) {
    logRequest('bypass', requestedModel, {})
    const errorMessage = `Model "${requestedModel}" is not configured. To use this model, either:

1. Add a model mapping to your API key: o3=opus-4.5:sk-ant-xxx
2. Add a default API key for OpenAI/ChatGPT fallback
3. Login via Sub Bridge OAuth to use your Claude/ChatGPT subscription
4. Use Codex CLI as backend: set OPENAI_BACKEND=codex-cli (or set API key to "codex-cli")

See https://github.com/buremba/sub-bridge for setup instructions.`

    logError(`Model not configured: ${requestedModel}`)
    return c.json({
      error: {
        message: errorMessage,
        type: 'invalid_request_error',
        code: 'model_not_configured',
      }
    }, 400)
  }

  const { claudeModel, apiKey: initialClaudeToken } = routing
  const claudeRefreshToken = oauthTokens?.claudeRefreshToken
  const usedOAuthClaude = Boolean(oauthTokens?.claudeToken && initialClaudeToken === oauthTokens.claudeToken)
  let claudeAccessToken = initialClaudeToken

  body.model = claudeModel

  if (body.input !== undefined && !body.messages) {
    if (typeof body.input === 'string') body.messages = [{ role: 'user', content: body.input }]
    else if (Array.isArray(body.input)) body.messages = body.input
    if (body.user && typeof body.user === 'string') body.messages = [{ role: 'system', content: body.user }, ...body.messages]
  }

  const systemMessages = body.messages?.filter((msg: any) => msg.role === 'system') || []
  body.messages = body.messages?.filter((msg: any) => msg.role !== 'system') || []

  if (body.messages.length === 0) {
    logError('No user messages in request')
    return c.json({ error: 'No messages provided' }, 400)
  }

  body.system = [
    { type: 'text', text: "You are Claude Code, Anthropic's official CLI for Claude." },
    { type: 'text', text: "[Proxied via Sub Bridge - user's Claude subscription]" },
    ...systemMessages.map((msg: any) => ({ type: 'text', text: msg.content || '' })),
    { type: 'text', text: "Remember: You are Claude (by Anthropic), powered by the Opus 4.5 model. If asked about your identity, you are Claude, not any other AI model." },
  ]

  const contextSize = JSON.stringify(body.messages || []).length
  const contextTokensEstimate = Math.ceil(contextSize / 4)
  const systemText = body.system.map((s: any) => s.text).join('\n')
  logRequest('claude', `${requestedModel} → ${claudeModel}`, {
    system: systemText, messages: body.messages, tools: body.tools, tokens: contextTokensEstimate
  })

  body.max_tokens = claudeModel.includes('opus') ? 32_000 : 64_000
  body.messages = convertMessages(body.messages)

  if (body.tools?.length) {
    body.tools = body.tools.map((tool: any, idx: number) => {
      let converted: any
      if (tool.type === 'function' && tool.function) {
        converted = { name: tool.function.name, description: tool.function.description || '', input_schema: tool.function.parameters || { type: 'object', properties: {} } }
      } else if (tool.name) {
        converted = { name: tool.name, description: tool.description || '', input_schema: tool.input_schema || tool.parameters || { type: 'object', properties: {} } }
      } else { converted = tool }
      if (idx === body.tools.length - 1) converted.cache_control = { type: 'ephemeral' }
      return converted
    })
  }

  if (body.tool_choice === 'auto') body.tool_choice = { type: 'auto' }
  else if (body.tool_choice === 'none' || body.tool_choice === null) delete body.tool_choice
  else if (body.tool_choice === 'required') body.tool_choice = { type: 'any' }
  else if (body.tool_choice?.function?.name) body.tool_choice = { type: 'tool', name: body.tool_choice.function.name }

  if (body.system.length > 0) body.system[body.system.length - 1].cache_control = { type: 'ephemeral' }

  const cleanBody: any = {}
  const allowedFields = ['model', 'messages', 'max_tokens', 'stop_sequences', 'stream', 'system', 'temperature', 'top_p', 'top_k', 'tools', 'tool_choice']
  for (const field of allowedFields) if (body[field] !== undefined) cleanBody[field] = body[field]

  // Context window validation and truncation
  const tokenEstimate = estimateRequestTokens(cleanBody)

  if (tokenEstimate.isOverLimit) {
    const effectiveLimit = Math.floor(CLAUDE_MAX_CONTEXT_TOKENS * (1 - SAFETY_MARGIN))

    if (contextOverflowMode === 'error') {
      return c.json({
        error: {
          message: `Context too large: estimated ${tokenEstimate.totalTokens.toLocaleString()} tokens exceeds ${CLAUDE_MAX_CONTEXT_TOKENS.toLocaleString()} limit. ` +
                   `Breakdown: system=${tokenEstimate.systemTokens.toLocaleString()}, messages=${tokenEstimate.messagesTokens.toLocaleString()}, tools=${tokenEstimate.toolsTokens.toLocaleString()}. ` +
                   `Enable context truncation with --context-overflow=truncate or reduce your conversation history.`,
          type: 'invalid_request_error',
          code: 'context_length_exceeded',
        }
      }, 400)
    }

    if (contextOverflowMode === 'truncate') {
      const truncationResult = truncateMessages(
        cleanBody.messages,
        effectiveLimit,
        tokenEstimate.systemTokens,
        tokenEstimate.toolsTokens
      )

      cleanBody.messages = truncationResult.messages

      if (truncationResult.truncationNotice && cleanBody.system) {
        // Add truncation notice to beginning of system prompt
        cleanBody.system = [
          { type: 'text', text: truncationResult.truncationNotice },
          ...cleanBody.system,
        ]
      }

      // Re-estimate and log the truncated context
      const newEstimate = estimateRequestTokens(cleanBody)
      logRequest('claude', `${requestedModel} → ${claudeModel} (truncated: -${truncationResult.removedCount} msgs, ~${newEstimate.totalTokens.toLocaleString()} tokens)`, {
        system: systemText,
        messages: cleanBody.messages,
        tools: cleanBody.tools,
        tokens: newEstimate.totalTokens,
      })
    } else if (contextOverflowMode === 'warn') {
      // Log warning but proceed (will likely fail at API level)
      logError(`Warning: Context estimated at ${tokenEstimate.totalTokens.toLocaleString()} tokens, may exceed ${CLAUDE_MAX_CONTEXT_TOKENS.toLocaleString()} limit`)
    }
  }

  const sendClaudeRequest = (token: string) => fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'authorization': `Bearer ${token}`,
      'anthropic-beta': 'oauth-2025-04-20,prompt-caching-2024-07-31',
      'anthropic-version': '2023-06-01',
      'accept': isStreaming ? 'text/event-stream' : 'application/json',
    },
    body: JSON.stringify(cleanBody),
  })

  let response = await sendClaudeRequest(claudeAccessToken)
  if (!response.ok && response.status === 401 && usedOAuthClaude && claudeRefreshToken) {
    try {
      const refreshed = await refreshClaudeToken(claudeRefreshToken)
      if (refreshed.accessToken) {
        claudeAccessToken = refreshed.accessToken
        response = await sendClaudeRequest(claudeAccessToken)
      }
    } catch {
      return c.json({
        error: {
          message: 'Claude OAuth refresh failed. Please re-authenticate.',
          type: 'authentication_error',
          code: 'authentication_error',
        }
      }, 401)
    }
  }

  if (!response.ok) {
    const errorText = await response.text()
    logError(errorText.slice(0, 200))

    // Try to parse the Anthropic error and convert to OpenAI format
    try {
      const anthropicError = JSON.parse(errorText)
      const errorMessage = anthropicError.error?.message || 'Unknown error'
      const errorType = anthropicError.error?.type || 'api_error'

      // Map Anthropic error types to user-friendly messages
      let userMessage = errorMessage
      if (errorType === 'rate_limit_error') {
        userMessage = `Rate limit exceeded: ${errorMessage}`
      } else if (errorType === 'authentication_error') {
        userMessage = `Authentication failed: ${errorMessage}`
      } else if (errorType === 'invalid_request_error') {
        userMessage = `Invalid request: ${errorMessage}`
      }

      // Return OpenAI-compatible error format
      const openAIError = {
        error: {
          message: userMessage,
          type: errorType,
          code: errorType,
        }
      }

      return c.json(openAIError, response.status as any)
    } catch (parseError) {
      // If parsing fails, return raw error
      return new Response(errorText, { status: response.status })
    }
  }

  logResponse(response.status)

  if (isStreaming) {
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    const converterState = createConverterState()
    return stream(c, async (s) => {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const results = processChunk(converterState, chunk, false)
        for (const result of results) {
          if (result.type === 'chunk') await s.write(`data: ${JSON.stringify(result.data)}\n\n`)
          else if (result.type === 'done') await s.write('data: [DONE]\n\n')
        }
      }
      reader.releaseLock()
    })
  } else {
    const responseData = await response.json()
    const openAIResponse = convertNonStreamingResponse(responseData as any)
    return c.json(openAIResponse)
  }
}

export function createChatRoutes() {
  const app = new Hono()

  // Base route for quick health checks
  app.get('/', (c) => c.json({ status: 'ok' }))

  // Models endpoint
  app.get('/models', async (c) => {
    const models: Array<{ id: string; object: 'model'; created: number; owned_by: string }> = []

    // Always include the Codex model for OpenAI-compatible backends.
    models.push({
      id: CHATGPT_DEFAULT_MODEL,
      object: 'model' as const,
      created: Math.floor(new Date('2025-01-01').getTime() / 1000),
      owned_by: 'openai',
    })

    // Best-effort: include Anthropic models (used for routing).
    try {
      const response = await fetch('https://models.dev/api.json')
      if (response.ok) {
        const modelsData = await response.json() as any
        const anthropicModels = modelsData.anthropic?.models || {}
        for (const [modelId, modelData] of Object.entries(anthropicModels) as Array<[string, any]>) {
          models.push({
            id: modelId,
            object: 'model' as const,
            created: Math.floor(new Date(modelData.release_date || '1970-01-01').getTime() / 1000),
            owned_by: 'anthropic',
          })
        }
      }
    } catch {
      // Ignore models.dev outages; Codex model is still returned.
    }

    return c.json({ object: 'list', data: models })
  })

  // Chat completions
  app.post('/chat/completions', (c) => handleChatCompletion(c))
  app.post('/messages', (c) => handleChatCompletion(c))

  return app
}
