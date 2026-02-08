import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'

export type CodexCliSandbox = 'read-only' | 'workspace-write' | 'danger-full-access'

export interface CodexCliRunOptions {
  cwd: string
  sandbox: CodexCliSandbox
  timeoutMs: number
  model?: string
}

export interface CodexCliUsage {
  input_tokens?: number
  cached_input_tokens?: number
  output_tokens?: number
  [key: string]: unknown
}

export interface CodexCliResult {
  threadId?: string
  text: string
  usage?: CodexCliUsage
}

function killProcess(child: ChildProcessWithoutNullStreams) {
  if (child.killed) return
  try { child.kill('SIGTERM') } catch { /* ignore */ }
  setTimeout(() => {
    try { child.kill('SIGKILL') } catch { /* ignore */ }
  }, 5_000).unref()
}

export async function runCodexCli(prompt: string, options: CodexCliRunOptions): Promise<CodexCliResult> {
  const args = [
    'exec',
    '-C',
    options.cwd,
    '--skip-git-repo-check',
    '--sandbox',
    options.sandbox,
    '--json',
  ]

  if (options.model) {
    args.push('--model', options.model)
  }

  // Read prompt from stdin to avoid shell/argv length limits.
  args.push('-')

  const child = spawn('codex', args, {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: process.env,
  })

  child.stdin.write(prompt)
  child.stdin.end()

  let stdoutBuf = ''
  let threadId: string | undefined
  let lastText = ''
  let usage: CodexCliUsage | undefined
  let errorMessage: string | undefined
  let stderrTail = ''

  const parseEventLine = (line: string) => {
    const trimmed = line.trim()
    if (!trimmed) return

    let evt: any
    try {
      evt = JSON.parse(trimmed)
    } catch {
      return
    }

    const type = evt?.type
    if (type === 'thread.started' && typeof evt.thread_id === 'string') {
      threadId = evt.thread_id
      return
    }

    if (type === 'item.completed' && evt.item) {
      const item = evt.item
      const itemType = item?.type
      if ((itemType === 'agent_message' || itemType === 'assistant_message') && typeof item.text === 'string') {
        lastText = item.text
      }
      return
    }

    if (type === 'turn.completed' && evt.usage && typeof evt.usage === 'object') {
      usage = evt.usage
      return
    }

    if (type === 'error' && typeof evt.message === 'string') {
      errorMessage = evt.message
      return
    }

    if (type === 'turn.failed' && typeof evt.error?.message === 'string') {
      errorMessage = evt.error.message
    }
  }

  child.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString('utf8')
    while (true) {
      const idx = stdoutBuf.indexOf('\n')
      if (idx === -1) break
      const line = stdoutBuf.slice(0, idx)
      stdoutBuf = stdoutBuf.slice(idx + 1)
      parseEventLine(line)
    }
  })

  child.stderr.on('data', (chunk) => {
    // Codex CLI can be noisy on stderr even on success; keep a small tail for debugging on failure.
    const s = chunk.toString('utf8')
    stderrTail = (stderrTail + s).slice(-32_000)
  })

  const timeout = setTimeout(() => {
    errorMessage = `Codex CLI timed out after ${options.timeoutMs}ms`
    killProcess(child)
  }, options.timeoutMs)
  timeout.unref()

  const exit = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    child.on('close', (code, signal) => resolve({ code, signal }))
  })

  clearTimeout(timeout)

  // Flush any remaining buffered stdout line.
  if (stdoutBuf.trim()) {
    parseEventLine(stdoutBuf)
  }

  if (exit.code !== 0) {
    const msg = errorMessage || stderrTail.trim() || `codex exited with code ${exit.code}${exit.signal ? ` (${exit.signal})` : ''}`
    throw new Error(msg)
  }

  if (!lastText) {
    if (errorMessage) throw new Error(errorMessage)
    throw new Error('Codex CLI produced no assistant message')
  }

  return {
    threadId,
    text: lastText,
    usage,
  }
}

