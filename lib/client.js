/**
 * @local/dsh-moss-precision — Client half.
 *
 * One compact control in `conversation.input.right`, ordered before the
 * existing 「推理等级」 entry, opening a dropdown with the three MOSS-Image
 * precision combos. Reads and writes go through the bundle's own same-origin
 * `/api/moss-precision` route; the Host half owns the restart.
 */
window.__ModuleLoader__.load({
  id: '@local/dsh-moss-precision',
  factory: (require) => {
    const React = require('react')
    const ReactDOM = require('react-dom')
    const h = React.createElement

    const NS = 'mossPrecision'
    const ENDPOINT = '/api/moss-precision'
    const PANEL_WIDTH = 340

    const zh = {
      'trigger.title': 'MOSS 精度组合',
      'trigger.aria': 'MOSS 精度组合，当前 {level}',
      'menu.title': 'MOSS 精度组合',
      'menu.hint': '完整权重不变 · 切换不下载、不替换文件',
      'level.bf16-bf16.name': 'DiT BF16 + TE BF16',
      'level.bf16-bf16.tag': '默认',
      'level.bf16-bf16.desc': '画质最好，内存占用最高',
      'level.fp8-bf16.name': 'DiT FP8 + TE BF16',
      'level.fp8-bf16.tag': '快速预览',
      'level.fp8-bf16.desc': '显存略省，画质略降',
      'level.fp8-fp8.name': 'DiT FP8 + TE FP8',
      'level.fp8-fp8.tag': '不推荐',
      'level.fp8-fp8.desc': '内存明显降低，画质有明显风险',
      'status.applying': '正在切换到 {level}…（Text Encoder 需要重启服务，约 1–2 分钟）',
      'status.applyingNoRestart': '正在切换到 {level}…（DiT 精度随下次请求生效，无需重启）',
      'status.running': '服务运行中 · 实际 TE {te}',
      'status.mismatch': '服务实际 TE 为 {te}，与所选组合不一致',
      'status.stopped': '服务未运行；选择会在下次启动时生效',
      'status.unknown': '无法读取服务状态',
      'status.error': '切换失败：{message}',
      'load.error': '无法读取精度状态',
      'value.none': '未知',
      'busy': '切换中…',
    }
    const en = {
      'trigger.title': 'MOSS precision',
      'trigger.aria': 'MOSS precision, currently {level}',
      'menu.title': 'MOSS precision',
      'menu.hint': 'Full weights stay on disk · switching never downloads or replaces files',
      'level.bf16-bf16.name': 'DiT BF16 + TE BF16',
      'level.bf16-bf16.tag': 'Default',
      'level.bf16-bf16.desc': 'Best quality, highest memory use',
      'level.fp8-bf16.name': 'DiT FP8 + TE BF16',
      'level.fp8-bf16.tag': 'Preview',
      'level.fp8-bf16.desc': 'Slightly less VRAM, slightly lower quality',
      'level.fp8-fp8.name': 'DiT FP8 + TE FP8',
      'level.fp8-fp8.tag': 'Not recommended',
      'level.fp8-fp8.desc': 'Clearly lower memory, visible quality risk',
      'status.applying': 'Switching to {level}… (Text Encoder needs a service restart, about 1–2 minutes)',
      'status.applyingNoRestart': 'Switching to {level}… (DiT precision applies to the next request, no restart)',
      'status.running': 'Service running · actual TE {te}',
      'status.mismatch': 'Service TE is {te}, which differs from the selected combo',
      'status.stopped': 'Service is not running; the choice applies on its next start',
      'status.unknown': 'Service state unavailable',
      'status.error': 'Switch failed: {message}',
      'load.error': 'Cannot read the precision state',
      'value.none': 'unknown',
      'busy': 'Switching…',
    }
    const DICT = { zh, en }

    /** Ids are stable; dtype details always come from the Host when it answers. */
    const FALLBACK_LEVELS = [{ id: 'bf16-bf16' }, { id: 'fp8-bf16' }, { id: 'fp8-fp8' }]

    function lang() {
      const value = typeof document === 'undefined' ? '' : document.documentElement.lang || ''
      return String(value).toLowerCase().startsWith('zh') ? 'zh' : 'en'
    }

    function translatorOf(bound) {
      const dict = DICT[lang()]
      return (key, params) => {
        let text
        if (typeof bound === 'function') {
          try { text = bound(key, params) } catch { text = undefined }
        }
        if (typeof text !== 'string' || text.length === 0 || text === key) text = dict[key] ?? key
        if (params !== undefined) {
          for (const [name, value] of Object.entries(params)) text = text.split(`{${name}}`).join(String(value))
        }
        return text
      }
    }

    function messageOf(error) {
      if (error === null || error === undefined) return undefined
      if (typeof error === 'string') return error
      if (typeof error.message === 'string' && error.message.length > 0) return error.message
      return String(error)
    }

    /** `torch.float8_e4m3fn` → `FP8 e4m3fn`; `torch.bfloat16` → `BF16`. */
    function shortDtype(value) {
      const text = String(value || '').replace(/^torch\./, '')
      if (text === 'bfloat16') return 'BF16'
      if (text === 'float32') return 'FP32'
      if (text === 'float16') return 'FP16'
      const fp8 = /^float8_(e[45]m[23](?:fn)?)$/.exec(text)
      if (fp8) return `FP8 ${fp8[1]}`
      return text || ''
    }

    // ------------------------------------------------------------ 样式
    const css = [
      ".dshMp_root{display:inline-flex;align-items:center;min-width:0}",
      ".dshMp_trigger{min-width:0;max-width:190px;height:28px;color:var(--dsw-alias-label-secondary,#b9b9b9);cursor:pointer;background:0 0;border:none;border-radius:24px;outline:none;align-items:center;gap:5px;padding:0 6px 0 8px;font-size:13px;font-weight:500;line-height:20px;display:flex}",
      ".dshMp_trigger:hover:not(:disabled){background:var(--dsw-alias-interactive-bg-hover,rgba(255,255,255,.08))}",
      ".dshMp_trigger:focus-visible{box-shadow:0 0 0 2px var(--dsw-alias-border-l3,rgba(255,255,255,.28))}",
      ".dshMp_trigger:disabled{color:var(--dsw-alias-label-dimmed,#7a7a7a);cursor:default}",
      ".dshMp_icon{color:var(--dsw-alias-brand-primary,#63e2a4);flex:none;display:block}",
      ".dshMp_name{letter-spacing:.02em;flex:none}",
      ".dshMp_chip{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 6px;border-radius:9px;font-size:11.5px;line-height:18px;color:var(--dsw-alias-brand-primary,#63e2a4);background:color-mix(in srgb,var(--dsw-alias-brand-primary,#63e2a4) 15%,transparent)}",
      ".dshMp_chipBusy{color:var(--dsw-alias-state-warn-primary,#e0a84e);background:color-mix(in srgb,var(--dsw-alias-state-warn-primary,#e0a84e) 16%,transparent)}",
      ".dshMp_chevron{color:var(--dsw-alias-label-caption,#8a8a8a);flex:none;transition:transform .12s}",
      ".dshMp_chevronOpen{transform:rotate(180deg)}",
      ".dshMp_panel{position:fixed;z-index:1100;box-sizing:border-box;padding:6px;border-radius:12px;background:var(--dsw-alias-bg-layer-1,#26262b);-webkit-backdrop-filter:blur(20px) saturate(1.15);backdrop-filter:blur(20px) saturate(1.15);border:1px solid var(--dsw-alias-border-l1,#3a3a3a);box-shadow:0 18px 44px rgba(0,0,0,.38),0 2px 8px rgba(0,0,0,.22);color:var(--dsw-alias-label-primary,#eee)}",
      ".dshMp_head{padding:6px 8px 8px}",
      ".dshMp_title{font-size:13px;font-weight:650;line-height:20px}",
      ".dshMp_hint{margin-top:1px;font-size:11.5px;line-height:16px;color:var(--dsw-alias-label-tertiary,#9a9a9a)}",
      ".dshMp_item{display:grid;grid-template-columns:14px minmax(0,1fr) auto;align-items:start;gap:8px;width:100%;box-sizing:border-box;padding:8px;border:none;border-radius:8px;background:0 0;color:inherit;text-align:left;cursor:pointer;outline:none}",
      ".dshMp_item:hover:not(:disabled){background:var(--dsw-alias-interactive-bg-hover,rgba(255,255,255,.07))}",
      ".dshMp_item:focus-visible{box-shadow:0 0 0 2px var(--dsw-alias-border-l3,rgba(255,255,255,.28))}",
      ".dshMp_item:disabled{cursor:default;opacity:.6}",
      ".dshMp_itemActive{background:color-mix(in srgb,var(--dsw-alias-brand-primary,#63e2a4) 11%,transparent)}",
      ".dshMp_radio{width:14px;height:14px;margin-top:2px;border-radius:50%;border:1.5px solid var(--dsw-alias-border-l2,#555);box-sizing:border-box;position:relative}",
      ".dshMp_radioOn{border-color:var(--dsw-alias-brand-primary,#63e2a4)}",
      ".dshMp_radioOn::after{content:'';position:absolute;inset:2.5px;border-radius:50%;background:var(--dsw-alias-brand-primary,#63e2a4)}",
      ".dshMp_itemBody{min-width:0;display:block}",
      ".dshMp_itemName{display:block;font-size:13px;line-height:19px;font-weight:550;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
      ".dshMp_itemDesc{display:block;font-size:11.5px;line-height:16px;color:var(--dsw-alias-label-tertiary,#9a9a9a);white-space:normal}",
      ".dshMp_itemMeta{display:block;margin-top:2px;font-size:11px;line-height:15px;color:var(--dsw-alias-label-caption,#8a8a8a);font-variant-numeric:tabular-nums}",
      ".dshMp_tag{justify-self:end;white-space:nowrap;font-size:10.5px;line-height:16px;padding:0 6px;border-radius:8px;color:var(--dsw-alias-label-tertiary,#9a9a9a);border:1px solid var(--dsw-alias-border-l1,#3a3a3a)}",
      ".dshMp_tagWarn{color:var(--dsw-alias-state-warn-primary,#e0a84e);border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary,#e0a84e) 45%,transparent)}",
      ".dshMp_footer{margin-top:4px;padding:7px 8px 4px;border-top:1px solid var(--dsw-alias-border-l1,#3a3a3a);font-size:11.5px;line-height:16px;color:var(--dsw-alias-label-tertiary,#9a9a9a);display:flex;gap:6px;align-items:flex-start}",
      ".dshMp_dot{flex:none;width:6px;height:6px;margin-top:5px;border-radius:50%;background:var(--dsw-alias-state-idle-primary,#7a7a7a)}",
      ".dshMp_dotOk{background:var(--dsw-alias-state-success-primary,#4ec98a)}",
      ".dshMp_dotWarn{background:var(--dsw-alias-state-warn-primary,#e0a84e)}",
      ".dshMp_dotError{background:var(--dsw-alias-state-error-primary,#e2685f)}",
      ".dshMp_footerText{min-width:0;white-space:normal}",
      ".dshMp_spinner{flex:none;width:11px;height:11px;margin-top:2px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:dshMpSpin .8s linear infinite}",
      "@keyframes dshMpSpin{to{transform:rotate(360deg)}}",
      "@media(prefers-reduced-motion:reduce){.dshMp_spinner{animation:none}}",
    ].join('')

    function ensureStyle() {
      const id = 'dsh-moss-precision/selector.css'
      if (typeof document === 'undefined') return
      if (document.querySelector(`style[data-plugin-css="${id}"]`) !== null) return
      const tag = document.createElement('style')
      tag.dataset.plugin = 'dsh-moss-precision'
      tag.dataset.pluginCss = id
      tag.textContent = css
      document.head.appendChild(tag)
    }

    // ------------------------------------------------------------ 数据面
    async function fetchState() {
      const response = await fetch(`${ENDPOINT}/state`, {
        cache: 'no-store',
        credentials: 'same-origin',
        headers: { accept: 'application/json' },
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok || payload === null || payload.ok !== true) {
        throw new Error((payload && payload.error) || `HTTP ${response.status}`)
      }
      return payload
    }

    async function postSelect(level) {
      const response = await fetch(`${ENDPOINT}/select`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'content-type': 'application/json', 'x-dsh-moss-precision': '1' },
        body: JSON.stringify({ level }),
      })
      const payload = await response.json().catch(() => null)
      if (payload === null) throw new Error(`HTTP ${response.status}`)
      return payload
    }

    function levelIds(snapshot) {
      const levels = snapshot && snapshot.data && snapshot.data.levels
      return Array.isArray(levels) && levels.length > 0
        ? levels.map((level) => level.id)
        : FALLBACK_LEVELS.map((level) => level.id)
    }

    function displayName(t, id, level) {
      const key = `level.${id}.name`
      const text = t(key)
      return text === key && level && level.label ? level.label : text
    }

    // ------------------------------------------------------------ 组件
    function MossPrecisionTrigger(props) {
      const t = React.useMemo(() => translatorOf(props.t), [props.t])
      const [snapshot, setSnapshot] = React.useState(null)
      const [open, setOpen] = React.useState(false)
      const [busy, setBusy] = React.useState(false)
      const [error, setError] = React.useState(null)
      const [panel, setPanel] = React.useState(null)
      const rootRef = React.useRef(null)
      const panelRef = React.useRef(null)
      const itemRefs = React.useRef([])
      const pending = React.useRef(null)

      const applying = (snapshot && snapshot.applying) || null
      const applyingLevel = applying && applying.level ? applying.level : null
      const service = (snapshot && snapshot.data && snapshot.data.service) || null
      const level = (snapshot && snapshot.data && snapshot.data.level) || null
      const levels = (snapshot && snapshot.data && snapshot.data.levels && snapshot.data.levels.length > 0)
        ? snapshot.data.levels
        : FALLBACK_LEVELS

      const load = React.useCallback(() => {
        fetchState().then((payload) => {
          setSnapshot(payload)
          setError(null)
        }, (reason) => {
          setError(messageOf(reason) || t('load.error'))
        })
      }, [t])

      React.useEffect(() => {
        ensureStyle()
        load()
      }, [load])

      // 切换中快速轮询，静默时只在打开面板时刷新。
      React.useEffect(() => {
        if (applyingLevel === null) {
          pending.current = null
          return undefined
        }
        const timer = setInterval(load, 2000)
        return () => clearInterval(timer)
      }, [applyingLevel, load])

      React.useEffect(() => {
        if (open) load()
      }, [open, load])

      // 面板定位：锚定按钮上沿，宽度固定，左右不越界。
      const place = React.useCallback(() => {
        if (rootRef.current === null) return
        const rect = rootRef.current.getBoundingClientRect()
        const left = Math.max(8, Math.min(rect.left, window.innerWidth - PANEL_WIDTH - 8))
        setPanel({ left, bottom: window.innerHeight - rect.top + 8 })
      }, [])

      React.useLayoutEffect(() => {
        if (!open) return undefined
        place()
        window.addEventListener('resize', place)
        window.addEventListener('scroll', place, true)
        return () => {
          window.removeEventListener('resize', place)
          window.removeEventListener('scroll', place, true)
        }
      }, [open, place])

      React.useEffect(() => {
        if (!open) return undefined
        const onPointerDown = (event) => {
          const target = event.target
          if (panelRef.current !== null && panelRef.current.contains(target)) return
          if (rootRef.current !== null && rootRef.current.contains(target)) return
          setOpen(false)
        }
        const onKeyDown = (event) => {
          if (event.key === 'Escape') {
            event.stopPropagation()
            setOpen(false)
          }
        }
        document.addEventListener('pointerdown', onPointerDown, true)
        document.addEventListener('keydown', onKeyDown, true)
        return () => {
          document.removeEventListener('pointerdown', onPointerDown, true)
          document.removeEventListener('keydown', onKeyDown, true)
        }
      }, [open])

      // 打开时把焦点放到当前档位，键盘用户可以直接上下选择。
      React.useEffect(() => {
        if (!open) return
        const ids = levelIds(snapshot)
        const index = Math.max(0, ids.indexOf(level))
        const node = itemRefs.current[index]
        if (node && typeof node.focus === 'function') node.focus({ preventScroll: true })
      }, [open, level])

      const onPanelKeyDown = (event) => {
        const nodes = itemRefs.current.filter(Boolean)
        const index = nodes.indexOf(document.activeElement)
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault()
          if (nodes.length === 0) return
          const delta = event.key === 'ArrowDown' ? 1 : -1
          const next = (Math.max(-1, index) + delta + nodes.length) % nodes.length
          nodes[next].focus({ preventScroll: true })
        } else if (event.key === 'Home' || event.key === 'End') {
          event.preventDefault()
          const target = event.key === 'Home' ? nodes[0] : nodes[nodes.length - 1]
          if (target) target.focus({ preventScroll: true })
        } else if (event.key === 'Tab') {
          setOpen(false)
        }
      }

      const choose = (id) => {
        if (busy || applyingLevel !== null) return
        if (id === level) {
          setOpen(false)
          return
        }
        const entry = levels.find((item) => item.id === id)
        // Remember whether this switch touches the Text Encoder so the footer
        // can say truthfully whether a restart is expected.
        pending.current = {
          level: id,
          teChanged: Boolean(entry && snapshot && snapshot.data && entry.te_dtype !== snapshot.data.te_dtype),
        }
        setBusy(true)
        setError(null)
        postSelect(id).then((payload) => {
          setBusy(false)
          if (!payload || payload.ok !== true) {
            setError((payload && payload.error) || t('status.unknown'))
            return
          }
          setSnapshot((current) => ({
            ...(current || { data: null }),
            applying: payload.applying || { level: id, startedAt: Date.now() },
            lastError: null,
          }))
        }, (reason) => {
          setBusy(false)
          setError(messageOf(reason) || t('status.unknown'))
        })
      }

      const levelName = level === null ? t('value.none') : displayName(t, level, levels.find((entry) => entry.id === level))
      const activeEntry = levels.find((entry) => entry.id === level) || {}
      const chipText = applyingLevel !== null
        ? t('busy')
        : activeEntry.short || (level === null ? t('value.none') : t(`level.${level}.tag`))

      const trigger = h('button', {
        type: 'button',
        ref: rootRef,
        className: 'dshMp_trigger',
        'aria-haspopup': 'menu',
        'aria-expanded': open,
        'aria-label': t('trigger.aria', { level: levelName }),
        title: t('trigger.title'),
        disabled: busy,
        onClick: () => {
          setError(null)
          setOpen((value) => !value)
        },
      }, [
        h('span', { key: 'icon', className: 'dshMp_icon', 'aria-hidden': true },
          h('svg', { width: 14, height: 14, viewBox: '0 0 16 16', fill: 'none' }, [
            h('rect', { key: 'a', x: 1.6, y: 2.2, width: 12.8, height: 3, rx: 1.5, fill: 'currentColor', opacity: 0.95 }),
            h('rect', { key: 'b', x: 1.6, y: 6.5, width: 9, height: 3, rx: 1.5, fill: 'currentColor', opacity: 0.6 }),
            h('rect', { key: 'c', x: 1.6, y: 10.8, width: 5.4, height: 3, rx: 1.5, fill: 'currentColor', opacity: 0.35 }),
          ])),
        h('span', { key: 'name', className: 'dshMp_name' }, 'MOSS'),
        h('span', {
          key: 'chip',
          className: applyingLevel !== null || busy ? 'dshMp_chip dshMp_chipBusy' : 'dshMp_chip',
        }, chipText),
        h('span', { key: 'chevron', className: open ? 'dshMp_chevron dshMp_chevronOpen' : 'dshMp_chevron', 'aria-hidden': true },
          h('svg', { width: 14, height: 14, viewBox: '0 0 16 16', fill: 'none' },
            h('path', { d: 'M4.2 6.3 8 10l3.8-3.7', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))),
      ])

      let footer
      if (error !== null) {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot dshMp_dotError' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.error', { message: error }))]
      } else if (applyingLevel !== null) {
        const noRestart = pending.current !== null
          && pending.current.level === applyingLevel
          && pending.current.teChanged === false
        footer = [
          h('span', { key: 'spin', className: 'dshMp_spinner', 'aria-hidden': true }),
          h('span', { key: 'text', className: 'dshMp_footerText' }, t(noRestart ? 'status.applyingNoRestart' : 'status.applying', { level: displayName(t, applyingLevel) })),
        ]
      } else if (service === null) {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.unknown'))]
      } else if (service.up !== true) {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.stopped'))]
      } else if (service.text_encoder_match === true) {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot dshMp_dotOk' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.running', { te: shortDtype(service.text_encoder_dtype) }))]
      } else if (service.text_encoder_match === false) {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot dshMp_dotWarn' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.mismatch', { te: shortDtype(service.text_encoder_dtype) }))]
      } else {
        footer = [h('span', { key: 'dot', className: 'dshMp_dot' }), h('span', { key: 'text', className: 'dshMp_footerText' }, t('status.unknown'))]
      }

      const items = levels.map((entry, index) => {
        const selected = entry.id === level
        const locked = busy || applyingLevel !== null
        return h('button', {
          key: entry.id,
          ref: (node) => { itemRefs.current[index] = node },
          type: 'button',
          role: 'menuitemradio',
          'aria-checked': selected,
          className: selected ? 'dshMp_item dshMp_itemActive' : 'dshMp_item',
          disabled: locked,
          onClick: () => choose(entry.id),
        }, [
          h('span', { key: 'radio', className: selected ? 'dshMp_radio dshMp_radioOn' : 'dshMp_radio', 'aria-hidden': true }),
          h('span', { key: 'body', className: 'dshMp_itemBody' }, [
            h('span', { key: 'name', className: 'dshMp_itemName' }, displayName(t, entry.id, entry)),
            h('span', { key: 'desc', className: 'dshMp_itemDesc' }, t(`level.${entry.id}.desc`)),
            entry.dit_dtype && entry.te_dtype
              ? h('span', { key: 'meta', className: 'dshMp_itemMeta' }, `DiT ${entry.dit_dtype} · TE ${entry.te_dtype}`)
              : null,
          ]),
          h('span', {
            key: 'tag',
            className: entry.id === 'fp8-fp8' ? 'dshMp_tag dshMp_tagWarn' : 'dshMp_tag',
          }, t(`level.${entry.id}.tag`)),
        ])
      })

      const menu = open && panel !== null ? ReactDOM.createPortal(
        h('div', {
          ref: panelRef,
          className: 'dshMp_panel',
          role: 'menu',
          'aria-label': t('menu.title'),
          style: { left: panel.left, bottom: panel.bottom, width: PANEL_WIDTH },
          onKeyDown: onPanelKeyDown,
        }, [
          h('div', { key: 'head', className: 'dshMp_head' }, [
            h('div', { key: 'title', className: 'dshMp_title' }, t('menu.title')),
            h('div', { key: 'hint', className: 'dshMp_hint' }, t('menu.hint')),
          ]),
          ...items,
          h('div', { key: 'footer', className: 'dshMp_footer' }, footer),
        ]),
        document.body,
      ) : null

      return h(React.Fragment, null, trigger, menu)
    }

    function MossPrecision(props) {
      return h(MossPrecisionTrigger, props)
    }

    const inject = ['slots', 'locale']

    function apply(ctx) {
      try {
        ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'dsh-moss-precision: dictionaries')
      } catch (error) {
        console.warn('[moss-precision] locale namespace unavailable:', error)
      }
      try {
        ctx.slots.inject('conversation.input.right', () => ctx.slots.register({
          name: 'conversation.input.right',
          id: 'moss-precision',
          // 推理等级（effort-trigger）order 0；负值保证 MOSS 按钮在它左侧。
          order: -10,
          locale: NS,
        }, MossPrecision))
      } catch (error) {
        console.warn('[moss-precision] seat registration failed:', error)
      }
    }

    return { apply, inject }
  },
})
