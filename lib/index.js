/**
 * @local/dsh-moss-precision — Host half.
 *
 * The MOSS-Image image service is controlled by two independent dials:
 *
 *   DiT precision   → `MOSSImageHFLoader.weight_dtype` inside every request
 *                     workflow (bf16 / fp8_e4m3fn). No restart, no extra file.
 *   Text Encoder    → a ComfyUI launch flag (`--bf16-text-enc` /
 *                     `--fp8_e4m3fn-text-enc`). Only a restart can change it.
 *
 * `tools/moss_precision.py` owns that mapping, the persisted choice, and the
 * restart. This file only exposes it to the Web page and to the agent; it keeps
 * no second copy of the level table.
 *
 * HTTP surface (same origin as the page, loopback only):
 *   GET  /api/moss-precision/state    current choice + live service state
 *   POST /api/moss-precision/select   {"level": "<id>"} — apply (may restart)
 */
import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

/** Required Host service: the Web page's HTTP carrier. */
export const inject = ['webServer']

const ROUTE = '/api/moss-precision'
const MARKER = 'x-dsh-moss-precision'
const MAX_BODY = 16 * 1024
const STATE_TIMEOUT_MS = 30_000
const APPLY_TIMEOUT_MS = 600_000

/** Level ids are the only part of the table this half needs; Python is authoritative. */
const LEVEL_IDS = ['bf16-bf16', 'fp8-bf16', 'fp8-fp8']

/**
 * Deployment root of the MOSS-Image ComfyUI installation.
 *
 * `MOSS_ROOT` wins so a profile can point at any checkout. Otherwise a local
 * `config.local.json` (`{"root": "..."}`) next to this package is used; it is
 * git-ignored and keeps machine-specific paths out of the published bundle.
 * @returns {string} absolute root, or '' when neither source is configured
 */
function mossRoot() {
  if (process.env.MOSS_ROOT) return process.env.MOSS_ROOT
  try {
    const local = JSON.parse(readFileSync(new URL('../config.local.json', import.meta.url), 'utf8'))
    if (typeof local.root === 'string' && local.root.trim().length > 0) return local.root.trim()
  } catch {
    // not configured locally
  }
  return ''
}

/**
 * Run `tools/moss_precision.py` and parse its JSON stdout.
 * @param {string[]} args positional flags for the script
 * @param {number} timeoutMs hard ceiling for the child
 * @returns {Promise<{ok: boolean, data?: object, error?: string, detail?: string}>}
 */
function runScript(args, timeoutMs) {
  return new Promise((resolve) => {
    const root = mossRoot()
    if (root === '') {
      resolve({ ok: false, error: '未配置 MOSS_ROOT 或 config.local.json（MOSS-Image 部署根目录）' })
      return
    }
    const python = process.env.MOSS_PYTHON || join(root, 'venv', 'Scripts', 'python.exe')
    const script = join(root, 'tools', 'moss_precision.py')
    let child
    try {
      child = spawn(python, [script, ...args], {
        cwd: root,
        windowsHide: true,
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
      })
    } catch (error) {
      resolve({ ok: false, error: `无法启动精度脚本：${String(error)}` })
      return
    }
    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (value) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve(value)
    }
    const timer = setTimeout(() => {
      try { child.kill() } catch { /* already gone */ }
      finish({ ok: false, error: `精度脚本超时（${Math.round(timeoutMs / 1000)} 秒）` })
    }, timeoutMs)
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('error', (error) => finish({ ok: false, error: String(error) }))
    child.on('close', (code) => {
      let data
      try { data = JSON.parse(stdout.trim()) } catch { data = undefined }
      const detail = (stderr || stdout).trim().slice(-3000)
      if (code !== 0) {
        finish({ ok: false, error: (data && data.error) || detail || `exit ${code}`, data, detail })
        return
      }
      if (data === undefined) {
        finish({ ok: false, error: detail || '精度脚本没有返回 JSON', detail })
        return
      }
      finish({ ok: true, data })
    })
  })
}

/**
 * Answer one request with JSON.
 * @param {import('node:http').ServerResponse} res
 * @param {number} status HTTP status
 * @param {unknown} payload JSON body
 */
function sendJson(res, status, payload) {
  if (res.writableEnded || res.destroyed) return
  let body
  try { body = JSON.stringify(payload) } catch { body = '{"ok":false}' }
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store',
  })
  res.end(body)
}

/**
 * Local-only + same-origin guard for the prefix route. The `/api` prefix is
 * matched before the authenticated bridge, so the route defends itself the way
 * other local plugins do: loopback Host, no cross-site fetch, matching Origin,
 * and a marker header on mutations.
 * @param {import('node:http').IncomingMessage} req
 * @param {boolean} mutation whether the method changes state
 * @returns {{status: number, error: string} | null}
 */
function requestProblem(req, mutation) {
  const host = typeof req.headers.host === 'string' ? req.headers.host : ''
  if (!/^(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/i.test(host)) {
    return { status: 403, error: 'forbidden host' }
  }
  if (req.headers['sec-fetch-site'] === 'cross-site') {
    return { status: 403, error: 'forbidden site' }
  }
  const origin = req.headers.origin
  if (typeof origin === 'string') {
    try {
      if (new URL(origin).host !== host) return { status: 403, error: 'forbidden origin' }
    } catch {
      return { status: 403, error: 'forbidden origin' }
    }
  }
  if (mutation && req.headers[MARKER] !== '1') {
    return { status: 403, error: 'missing client marker' }
  }
  if (mutation) {
    const type = String(req.headers['content-type'] || '').split(';', 1)[0].trim().toLowerCase()
    if (type !== 'application/json') return { status: 415, error: 'content-type must be application/json' }
  }
  return null
}

/**
 * Read a bounded JSON body.
 * @param {import('node:http').IncomingMessage} req
 * @returns {Promise<string | null>} body text, or null when too large
 */
function readBody(req) {
  return new Promise((resolve) => {
    let size = 0
    const chunks = []
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > MAX_BODY) {
        resolve(null)
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', () => resolve(null))
  })
}

/** Short model-facing summary of one state snapshot. */
function summarize(state) {
  const service = state.service || {}
  return {
    level: state.level,
    label: state.label,
    dit_dtype: state.dit_dtype,
    text_encoder_dtype: state.te_dtype,
    service_running: service.up === true,
    service_text_encoder_dtype: service.text_encoder_dtype ?? null,
    effective: service.up === true && service.text_encoder_match === true,
  }
}

/**
 * Register the local HTTP surface and the agent tool.
 * @param {import('@deepseek-ai/cordis').Context} ctx
 */
export function apply(ctx) {
  const webServer = ctx.webServer
  if (!webServer || typeof webServer.register !== 'function') {
    console.warn('[moss-precision] webServer unavailable; the selector route is not served')
    return
  }

  /** In-flight switch, shown by the page while the service restarts. */
  let applying = null
  /** The in-flight `--set` child, shared by the tool and the HTTP route. */
  let applyPromise = null
  /** Last failed switch, surfaced in the page instead of being swallowed. */
  let lastError = null

  const state = async () => {
    const result = await runScript(['--state'], STATE_TIMEOUT_MS)
    if (!result.ok || !result.data) {
      return { ok: false, error: result.error || '无法读取精度状态', applying, lastError }
    }
    return { ok: true, data: result.data, applying, lastError }
  }

  /** Kick off a switch without waiting for the restart; the page polls `state`. */
  const beginApply = (level) => {
    if (applying !== null) return null
    applying = { level, startedAt: Date.now() }
    lastError = null
    applyPromise = runScript(['--set', level, '--json', '--by', 'ui'], APPLY_TIMEOUT_MS)
      .then((result) => {
        const data = result.data
        if (!result.ok || !data || data.ok !== true) {
          lastError = (data && data.error) || result.error || '切换失败'
        }
        applying = null
        applyPromise = null
        return result
      })
    return applyPromise
  }

  ctx.effect(() => webServer.register({
    kind: 'prefix',
    path: ROUTE,
    handler: async (req, res) => {
      const problem = requestProblem(req, req.method === 'POST')
      if (problem !== null) {
        sendJson(res, problem.status, { ok: false, error: problem.error })
        return
      }
      let path = ROUTE
      try { path = new URL(req.url || '/', 'http://localhost').pathname.replace(/\/+$/, '') } catch { /* keep */ }
      try {
        if (req.method === 'GET' && path === `${ROUTE}/state`) {
          const snapshot = await state()
          sendJson(res, snapshot.ok ? 200 : 503, snapshot)
          return
        }
        if (req.method === 'POST' && path === `${ROUTE}/select`) {
          const body = await readBody(req)
          if (body === null) {
            sendJson(res, 413, { ok: false, error: '请求体过大' })
            return
          }
          let parsed
          try { parsed = JSON.parse(body) } catch { parsed = undefined }
          const level = parsed && typeof parsed.level === 'string' ? parsed.level : ''
          if (!LEVEL_IDS.includes(level)) {
            sendJson(res, 400, { ok: false, error: `未知精度组合：${level || '(空)'}` })
            return
          }
          // The restart can take minutes: accept, then let the page poll `state`.
          const promise = beginApply(level)
          if (promise === null) {
            sendJson(res, 409, { ok: false, error: `正在切换到 ${applying.level}，请稍候`, applying })
            return
          }
          sendJson(res, 202, { ok: true, applying })
          return
        }
        sendJson(res, 404, { ok: false, error: 'not found' })
      } catch (error) {
        console.warn('[moss-precision] request failed:', error)
        sendJson(res, 500, { ok: false, error: String(error && error.message ? error.message : error) })
      }
    },
  }), 'dsh-moss-precision: precision route')

  // Optional: profiles without the tool registry still serve the page control.
  ctx.inject(['tools'], (scope) => {
    scope.effect(() => scope.tools.register({
      name: 'moss_precision',
      description: 'Read or change the MOSS-Image precision combo used by the local image service. '
        + 'bf16-bf16 = DiT BF16 + Text Encoder BF16 (default, best quality); '
        + 'fp8-bf16 = DiT FP8 + Text Encoder BF16 (quick preview); '
        + 'fp8-fp8 = both FP8 (lowest memory, visible quality risk, not recommended). '
        + 'Switching never downloads or replaces weights; changing the Text Encoder precision restarts the local ComfyUI service. '
        + 'Only change it when the user explicitly asks.',
      parameters: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['get', 'set'], description: '`get` current combo, `set` a new one.' },
          level: { type: 'string', enum: LEVEL_IDS, description: 'Required for `set`.' },
        },
        required: ['action'],
        additionalProperties: false,
      },
      output: {
        schema: {},
        render: (_args, value) => [{ type: 'text', text: value && value.ok === false ? `MOSS precision: ${value.error}` : JSON.stringify(value) }],
      },
      presentCall: (args) => ({ card: 'generic', title: 'MOSS precision', kind: 'edit', rawInput: args && args.level ? String(args.level) : 'get' }),
      async execute(args) {
        if (args.action === 'get') {
          const snapshot = await state()
          if (!snapshot.ok) return { ok: false, error: snapshot.error }
          return { ok: true, ...summarize(snapshot.data) }
        }
        if (!LEVEL_IDS.includes(args.level)) {
          return { ok: false, error: '设置精度组合时必须提供 level（bf16-bf16 / fp8-bf16 / fp8-fp8）' }
        }
        // Do not block the tool call on a 1–2 minute service restart: start it
        // and let the caller confirm with `get`.
        const promise = beginApply(args.level)
        if (promise === null) {
          return { ok: false, error: `正在切换到 ${applying.level}，请稍候` }
        }
        return {
          ok: true,
          started: true,
          level: args.level,
          note: '切换已开始；涉及 Text Encoder 时会重启本地 ComfyUI（约 1–2 分钟）。之后用 action=get 查看是否已生效。',
        }
      },
      }), 'dsh-moss-precision: agent tool')
  })
}
