/**
 * @local/dsh-moss-precision — Prompt-enhance half.
 *
 * Compiles the composer draft with the bundled `moss-image-prompt-skill`
 * through the harness LLM service (`ctx.llm.stream`). The browser menu item
 * only reads/writes the draft; every model-facing decision lives here, so the
 * route stays testable with a structural `llm` stub.
 *
 * The skill is an instruction document, not an executable library: the
 * "enhancement capability" is exactly "send SKILL.md as the system prompt and
 * let the model compile the draft by its rules". No provider or model name is
 * invented — the route is resolved from the live session, the harness default
 * model, and the legacy settings entry, in that order.
 */
import { readFileSync } from 'node:fs'

/** Same-origin host route served under the existing precision prefix. */
export const ENHANCE_ROUTE = '/api/moss-precision/enhance'
/** Longest accepted draft (code points). The prompt is a single composer draft. */
export const MAX_INPUT_CHARS = 8000
/** One manual compile; the skill asks for a long answer, so allow a slow model. */
export const ENHANCE_TIMEOUT_MS = 180_000
/** Enough for Positive + Negative + sampling parameters. */
const MAX_OUTPUT_TOKENS = 4096
const TEMPERATURE = 0.35
const SKILL_URL = new URL('../prompt/moss-image-prompt-skill.md', import.meta.url)

/**
 * Task frame appended to the skill. The skill is written as a standing
 * instruction set; this block tells the model that this call is one concrete
 * compile request and fixes the output contract the composer replacement
 * depends on (plain text, Positive + Negative + parameters).
 */
const TASK_FRAME = `
## 本次任务（DSH 插件一次性编译请求）

你现在收到的不是普通对话，而是一次一次性的提示词编译请求：
- 用户消息 <raw_input> 里是原始提示词 / Danbooru tags / 模糊创作需求。
- 严格按本文件 §2 → §4 → §5 → §6 → §7 → §8/§9/§9.1/§10 → §11 的顺序编译，不要跳步。
- 最终只输出一套可直接粘贴使用的 MOSS-Image 提示词包，且必须包含三部分：
  1. Positive Prompt（按 §5 选择分块或单段短文，按 §11 模板组织）；
  2. Negative Prompt（首排焊死 §0 的着装与姿态封杀组，再追加场景专项封杀词）；
  3. 采样参数（分辨率 / Steps / CFG / Shift / Sampler / SLG；按 §0 的快速参数，文字类画面按 §9 调整）。
- 输出纯文本：不要代码围栏，不要解释，不要复述原文，不要「以下是……」之类的前言。
- 保留原始创作意图；缺失细节用简短占位符（如「待补充：…」）标注，不要编造具体人名、品牌或作品名。
- 输出语言与原文一致；Positive / Negative 正文跟随原文语言，负面池保留规范要求的英文封杀词。
`.trim()

/** One prompt-enhance failure with a stable code for HTTP status mapping. */
export class PromptEnhanceError extends Error {
  /**
   * @param {'empty'|'too-long'|'unavailable'|'unconfigured'|'timeout'|'aborted'|'truncated'|'upstream'} code
   * @param {string} message displayable text; the draft is never touched
   */
  constructor(code, message) {
    super(message)
    this.name = 'PromptEnhanceError'
    this.code = code
  }
}

/** Best-effort error text. */
function messageOf(error) {
  if (error === null || error === undefined) return ''
  if (typeof error === 'string') return error
  if (typeof error.message === 'string' && error.message !== '') return error.message
  return String(error)
}

/** HTTP status for one failure code. */
export function statusOf(code) {
  if (code === 'empty' || code === 'too-long') return 422
  if (code === 'timeout') return 504
  return 502
}

/** Cache the bundled skill after the first request; a missing file fails that request only. */
let cachedSystemPrompt = null

/**
 * Read the bundled `moss-image-prompt-skill` and append the one-shot task frame.
 * @returns {string} the system prompt for one compile call
 * @throws {PromptEnhanceError} when the bundled skill cannot be read
 */
export function skillSystemPrompt() {
  if (cachedSystemPrompt !== null) return cachedSystemPrompt
  let raw
  try {
    raw = readFileSync(SKILL_URL, 'utf8')
  } catch (error) {
    throw new PromptEnhanceError('unavailable', `无法读取内置的 moss-image-prompt-skill：${messageOf(error)}`)
  }
  if (raw.trim() === '') {
    throw new PromptEnhanceError('unavailable', '内置的 moss-image-prompt-skill 为空')
  }
  cachedSystemPrompt = `${raw.trimEnd()}\n\n${TASK_FRAME}\n`
  return cachedSystemPrompt
}

/**
 * Frame one draft as literal data. A literal `</raw_input>` inside the draft is
 * neutralized so the delimiter cannot be closed early.
 * @param {string} text
 * @returns {string}
 */
export function frameUserPrompt(text) {
  const safe = String(text).replace(/<\/?raw_input>/gi, (tag) => tag.replace('<', '<\\'))
  return `请编译下面 <raw_input> 中的原始提示词：\n\n<raw_input>\n${safe}\n</raw_input>`
}

/**
 * Remove a model-supplied outer code fence and trim blank edges. The skill
 * asks for plain text, but a fence is a common harmless slip; the draft
 * replacement must contain only the prompt package.
 * @param {unknown} raw
 * @returns {string}
 */
export function normalizeOutput(raw) {
  let text = String(raw ?? '').replace(/\r\n/g, '\n').trim()
  const fenced = /^```[^\n]*\n([\s\S]*?)\n```$/.exec(text)
  if (fenced !== null) return fenced[1].trim()
  text = text.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '')
  return text.trim()
}

/** Narrow an untrusted provider/model pair. */
function routeOf(value) {
  if (value === null || value === undefined || typeof value !== 'object') return undefined
  const provider = typeof value.provider === 'string' ? value.provider.trim() : ''
  const model = typeof value.model === 'string' ? value.model.trim() : ''
  return provider !== '' && model !== '' ? { provider, model } : undefined
}

/** Read one cordis service without letting a cordis-level throw escape. */
function serviceOf(ctx, name) {
  try {
    return ctx.get(name)
  } catch {
    return undefined
  }
}

/** The session's logged request route, when the session store can answer. */
function sessionRouteOf(ctx, sessionId) {
  if (typeof sessionId !== 'string' || sessionId === '') return undefined
  const sessions = serviceOf(ctx, 'sessions')
  if (sessions === null || sessions === undefined || typeof sessions.get !== 'function') return undefined
  let session
  try {
    session = sessions.get(sessionId)
  } catch {
    return undefined
  }
  if (session === null || session === undefined) return undefined
  const header = session.requestHeader
  let epoch
  try {
    epoch = typeof header === 'function' ? header.call(session) : header
  } catch {
    return undefined
  }
  return routeOf(epoch && epoch.config)
}

/** The harness-wide default model, across the service shapes this host line exposes. */
function defaultRouteOf(ctx) {
  const byService = serviceOf(ctx, 'agentDefaultModel')
  if (byService !== null && byService !== undefined && typeof byService.currentSelection === 'function') {
    try {
      const route = routeOf(byService.currentSelection())
      if (route !== undefined) return route
    } catch {
      // fall through to the legacy settings read
    }
  }
  const settings = serviceOf(ctx, 'settings')
  if (settings !== null && settings !== undefined && typeof settings.get === 'function') {
    try {
      return routeOf(settings.get('agent-default-model'))
    } catch {
      // no answer
    }
  }
  return undefined
}

/**
 * Resolve the model route by precedence: the session's last request route,
 * then the harness default selection. Returns undefined when no source names
 * one — the caller reports `unconfigured` instead of guessing a model.
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {string|undefined} sessionId
 * @returns {{provider: string, model: string}|undefined}
 */
export function resolveRoute(ctx, sessionId) {
  return sessionRouteOf(ctx, sessionId) ?? defaultRouteOf(ctx)
}

/** Reject a promise when the signal aborts, without leaking the listener. */
function withAbort(promise, signal) {
  if (signal.aborted) return Promise.reject(abortError())
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortError())
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort)
        resolve(value)
      },
      (error) => {
        signal.removeEventListener('abort', onAbort)
        reject(error)
      },
    )
  })
}

/** Signal-shaped error, matched the same way the harness does. */
function abortError() {
  const error = new Error('moss-prompt-enhance: aborted')
  error.name = 'AbortError'
  return error
}

/** Pull visible text out of the raw stream chunks. Mirrors dsh-llm's vocabulary. */
function collectChunk(chunk, state) {
  if (chunk === null || typeof chunk !== 'object') return
  if (chunk.type === 'text-delta' && typeof chunk.text === 'string') {
    state.text += chunk.text
    return
  }
  if (chunk.type === 'finish') state.finish = chunk.reason
}

/**
 * Compile one draft through `ctx.llm.stream`.
 * @param {object} options
 * @param {import('@deepseek-ai/cordis').Context} options.ctx
 * @param {string} options.text already validated, non-empty draft
 * @param {string} [options.sessionId]
 * @param {AbortSignal} [options.signal]
 * @param {number} [options.timeoutMs]
 * @returns {Promise<{text: string, provider: string, model: string, elapsedMs: number}>}
 * @throws {PromptEnhanceError}
 */
export async function enhancePrompt({ ctx, text, sessionId, signal, timeoutMs = ENHANCE_TIMEOUT_MS }) {
  const started = Date.now()
  const llm = serviceOf(ctx, 'llm')
  if (llm === null || llm === undefined || typeof llm.stream !== 'function') {
    throw new PromptEnhanceError('unavailable', '宿主 LLM 服务不可用，无法调用 moss-image-prompt-skill')
  }
  const route = resolveRoute(ctx, sessionId)
  if (route === undefined) {
    throw new PromptEnhanceError('unconfigured', '尚未确定用于增强的模型：请先在当前会话发送一条消息，或在设置中选择默认模型')
  }
  const system = skillSystemPrompt()

  const controller = new AbortController()
  let timedOut = false
  let callerAborted = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const onCallerAbort = () => {
    if (!timedOut) callerAborted = true
    controller.abort()
  }
  if (signal !== undefined) {
    signal.addEventListener('abort', onCallerAbort, { once: true })
    if (signal.aborted) onCallerAbort()
  }

  const streamSignal = controller.signal
  const state = { text: '', finish: undefined }
  let iterator
  let exhausted = false
  try {
    streamSignal.throwIfAborted()
    iterator = llm.stream({
      provider: route.provider,
      model: route.model,
      system,
      messages: [{ role: 'user', content: [{ type: 'text', text: frameUserPrompt(text) }] }],
      temperature: TEMPERATURE,
      maxTokens: MAX_OUTPUT_TOKENS,
      signal: streamSignal,
      ...(typeof sessionId === 'string' && sessionId !== '' ? { sessionId } : {}),
    })[Symbol.asyncIterator]()
    while (true) {
      streamSignal.throwIfAborted()
      const next = await withAbort(iterator.next(), streamSignal)
      if (next.done) {
        exhausted = true
        break
      }
      collectChunk(next.value, state)
    }
    if (timedOut) throw new PromptEnhanceError('timeout', `提示词增强超时（${Math.round(timeoutMs / 1000)} 秒），原输入未改动`)
    if (callerAborted) throw new PromptEnhanceError('aborted', '提示词增强已取消，原输入未改动')
    const finish = state.finish
    if (finish !== null && typeof finish === 'object') {
      if (finish.kind === 'error') {
        const failure = finish.failure && typeof finish.failure === 'object' ? finish.failure : {}
        const detail = messageOf(failure.message) || messageOf(failure.code) || '未知错误'
        throw new PromptEnhanceError('upstream', `模型服务返回错误：${detail}`)
      }
      if (finish.kind === 'aborted') {
        throw new PromptEnhanceError('aborted', '提示词增强已中断，原输入未改动')
      }
      if (finish.kind === 'max-tokens') {
        throw new PromptEnhanceError('truncated', '增强结果达到输出上限，请精简输入后重试；原输入未改动')
      }
    }
    const normalized = normalizeOutput(state.text)
    if (normalized === '') {
      throw new PromptEnhanceError('upstream', '模型返回了空结果，请重试；原输入未改动')
    }
    return {
      text: normalized,
      provider: route.provider,
      model: route.model,
      elapsedMs: Date.now() - started,
    }
  } catch (error) {
    if (error instanceof PromptEnhanceError) throw error
    if (timedOut) throw new PromptEnhanceError('timeout', `提示词增强超时（${Math.round(timeoutMs / 1000)} 秒），原输入未改动`)
    if (callerAborted || (error !== null && typeof error === 'object' && error.name === 'AbortError')) {
      throw new PromptEnhanceError('aborted', '提示词增强已取消，原输入未改动')
    }
    throw new PromptEnhanceError('upstream', `提示词增强失败：${messageOf(error)}`)
  } finally {
    clearTimeout(timer)
    if (signal !== undefined) signal.removeEventListener('abort', onCallerAbort)
    // On timeout/cancel the pending next() may never settle; ask the underlying
    // stream to finalize its connection instead of leaving it to the GC.
    if (!exhausted && iterator !== undefined && typeof iterator.return === 'function') {
      void Promise.resolve(iterator.return()).catch(() => {})
    }
  }
}
