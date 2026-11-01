import { useEffect, useRef, useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import Plotly from 'plotly.js-dist-min'
import { api, EvalResult, Method } from './api'

const TABS = ['Calculadora', 'Física', 'Programação', 'Engenharia numérica', 'Gráficos', 'Histórico', 'Configurações'] as const
type Tab = (typeof TABS)[number]
const MODULE_OF: Partial<Record<Tab, string>> = { 'Física': 'physics', 'Programação': 'programming', 'Engenharia numérica': 'numeric' }
const EXAMPLES = ['integrate(x^2, x)', 'differentiate(sin(x)*x^2, x)', 'limit(sin(x)/x, x, 0)', 'solve(x^2 - 5*x + 6 = 0, x)',
  'det(matrix([[1,2],[3,4]]))', 'solve(x^2 - 4 < 0, x)']
const card = 'rounded-lg border border-zinc-200 dark:border-zinc-800 p-4'
const input = 'w-full rounded border border-zinc-300 dark:border-zinc-700 bg-transparent px-2 py-1 font-mono text-sm'
const btn = 'rounded bg-indigo-600 px-3 py-1 text-sm text-white hover:bg-indigo-500 disabled:opacity-50'

type HistItem = { expr: string; result: string; latex: string; t: number }
const loadHist = (): HistItem[] => { try { return JSON.parse(localStorage.getItem('mathlab.history') || '[]') } catch { return [] } }

function Tex({ tex }: { tex: string }) {
  return <span dangerouslySetInnerHTML={{ __html: katex.renderToString(tex, { throwOnError: false, displayMode: true }) }} />
}
function Err({ msg }: { msg: string }) { return msg ? <p className="mt-2 text-sm text-red-500">{msg}</p> : null }

function Calculator({ onResult }: { onResult: (h: HistItem) => void }) {
  const [expr, setExpr] = useState('integrate(x^2, x)')
  const [res, setRes] = useState<EvalResult | null>(null)
  const [err, setErr] = useState(''); const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true); setErr('')
    try { const r = await api.evaluate(expr); setRes(r); onResult({ expr, result: r.result, latex: r.latex, t: Date.now() }) }
    catch (e) { setRes(null); setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="space-y-3">
      <div className={card}>
        <textarea className={input + ' h-24'} value={expr} onChange={e => setExpr(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run() }} />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button className={btn} disabled={busy} onClick={run}>Calcular (Ctrl+Enter)</button>
          {EXAMPLES.map(x => <button key={x} className="rounded border border-zinc-300 dark:border-zinc-700 px-2 py-0.5 text-xs font-mono" onClick={() => setExpr(x)}>{x}</button>)}
        </div>
        <Err msg={err} />
      </div>
      {res && <div className={card}>
        <div className="flex gap-2 text-xs">
          <span className="rounded bg-zinc-200 dark:bg-zinc-800 px-2 py-0.5">{res.exact ? 'Exato' : 'Aproximado'}</span>
          {res.verified !== null && <span className="rounded bg-zinc-200 dark:bg-zinc-800 px-2 py-0.5">{res.verified ? 'Verificado ✓' : 'Verificação falhou'}</span>}
        </div>
        <Tex tex={res.latex} />
        {res.note && <p className="text-sm text-zinc-500">{res.note}</p>}
        {res.steps.length > 0 && <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm">
          {res.steps.map((s, i) => <li key={i}><b>{s.title}</b><Tex tex={s.latex} /></li>)}</ol>}
      </div>}
    </div>
  )
}

function ModuleRunner({ module, methods }: { module: string; methods: Method[] }) {
  const [name, setName] = useState(methods[0]?.name ?? '')
  const [vals, setVals] = useState<Record<string, string>>({})
  const [out, setOut] = useState<any>(null); const [err, setErr] = useState('')
  const m = methods.find(x => x.name === name)
  useEffect(() => { setName(methods[0]?.name ?? ''); setVals({}); setOut(null); setErr('') }, [module])
  const parse = (s: string) => { try { return JSON.parse(s) } catch { return s } }
  const run = async () => {
    setErr(''); setOut(null)
    const params: Record<string, unknown> = {}
    m?.params.forEach(p => { const v = (vals[p.name] ?? '').trim(); if (v !== '') params[p.name] = parse(v) })
    try { setOut(await api.run(module, name, params)) } catch (e) { setErr((e as Error).message) }
  }
  if (!m) return <p>Carregando…</p>
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div className={card}>
        <select className={input} value={name} onChange={e => { setName(e.target.value); setVals({}); setOut(null) }}>
          {methods.map(x => <option key={x.name} value={x.name}>{x.name}</option>)}</select>
        <p className="my-2 text-sm text-zinc-500">{m.doc}</p>
        {m.params.map(p => (
          <label key={p.name} className="mb-2 block text-xs">{p.name}{p.required ? ' *' : ` (padrão: ${JSON.stringify(p.default)})`}
            <input className={input} value={vals[p.name] ?? ''} onChange={e => setVals({ ...vals, [p.name]: e.target.value })}
              placeholder="número, texto ou lista JSON, ex.: [[1,2],[3,4]]" /></label>))}
        <button className={btn} onClick={run}>Calcular</button>
        <Err msg={err} />
      </div>
      <div className={card}>
        {out?.formula && <Tex tex={out.formula} />}
        {out?.values && <table className="my-2 text-sm"><tbody>{Object.entries<any>(out.values).map(([k, v]) =>
          <tr key={k}><td className="pr-3 font-mono">{k}</td><td className="pr-2">{v.value}</td><td className="text-zinc-500">{v.unit}</td></tr>)}</tbody></table>}
        <pre className="overflow-auto text-xs">{out ? JSON.stringify(out, null, 2) : 'O resultado aparece aqui.'}</pre>
      </div>
    </div>
  )
}

function Graphs({ dark }: { dark: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const [fns, setFns] = useState('sin(x); x^2/10 - 1'); const [xmin, setMin] = useState('-6'); const [xmax, setMax] = useState('6')
  const [err, setErr] = useState('')
  const draw = async () => {
    setErr('')
    try {
      const r = await api.plot({ expressions: fns.split(';').map(s => s.trim()).filter(Boolean), xmin: +xmin, xmax: +xmax })
      const traces: any[] = r.series.map((s: any) => ({ x: r.x, y: s.y, mode: 'lines', name: s.expression, connectgaps: false }))
      r.series.forEach((s: any) => { if (s.zeros_approx.length) traces.push({ x: s.zeros_approx, y: s.zeros_approx.map(() => 0),
        mode: 'markers', name: `zeros de ${s.expression} (aprox.)`, marker: { size: 8 } }) })
      Plotly.react(ref.current, traces, { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: dark ? '#e4e4e7' : '#18181b' }, margin: { t: 20 }, xaxis: { gridcolor: dark ? '#3f3f46' : '#e4e4e7' },
        yaxis: { gridcolor: dark ? '#3f3f46' : '#e4e4e7' } }, { responsive: true })
    } catch (e) { setErr((e as Error).message) }
  }
  useEffect(() => { draw() }, [dark])
  return (
    <div className="space-y-3">
      <div className={card + ' grid gap-2 md:grid-cols-[1fr_6rem_6rem_auto]'}>
        <input className={input} value={fns} onChange={e => setFns(e.target.value)} title="Funções de x separadas por ;" />
        <input className={input} value={xmin} onChange={e => setMin(e.target.value)} /><input className={input} value={xmax} onChange={e => setMax(e.target.value)} />
        <button className={btn} onClick={draw}>Plotar</button>
        <Err msg={err} />
      </div>
      <div className={card}><div ref={ref} style={{ height: 480 }} /></div>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('Calculadora')
  const [dark, setDark] = useState(localStorage.getItem('mathlab.theme') !== 'light')
  const [hist, setHist] = useState<HistItem[]>(loadHist)
  const [mods, setMods] = useState<Record<string, Method[]>>({}); const [modErr, setModErr] = useState('')
  useEffect(() => { document.documentElement.classList.toggle('dark', dark); localStorage.setItem('mathlab.theme', dark ? 'dark' : 'light') }, [dark])
  useEffect(() => { api.modules().then(setMods).catch(e => setModErr(`Backend indisponível: ${e.message}`)) }, [])
  const addHist = (h: HistItem) => { const n = [h, ...hist].slice(0, 200); setHist(n); localStorage.setItem('mathlab.history', JSON.stringify(n)) }
  const mod = MODULE_OF[tab]
  return (
    <div className="mx-auto max-w-6xl p-4">
      <header className="mb-4 flex items-center justify-between"><h1 className="text-xl font-semibold">MathLab</h1>
        <button className="text-sm underline" onClick={() => setDark(!dark)}>{dark ? 'Tema claro' : 'Tema escuro'}</button></header>
      <nav className="mb-4 flex flex-wrap gap-1">{TABS.map(t =>
        <button key={t} onClick={() => setTab(t)} className={`rounded px-3 py-1 text-sm ${t === tab ? 'bg-indigo-600 text-white' : 'bg-zinc-200 dark:bg-zinc-800'}`}>{t}</button>)}</nav>
      <Err msg={modErr} />
      {tab === 'Calculadora' && <Calculator onResult={addHist} />}
      {mod && mods[mod] && <ModuleRunner module={mod} methods={mods[mod]} />}
      {tab === 'Gráficos' && <Graphs dark={dark} />}
      {tab === 'Histórico' && <div className={card}>{hist.length === 0 ? 'Sem histórico.' : hist.map(h =>
        <div key={h.t} className="border-b border-zinc-200 py-2 dark:border-zinc-800"><code className="text-xs">{h.expr}</code><Tex tex={h.latex} /></div>)}
        {hist.length > 0 && <button className="mt-2 text-sm underline" onClick={() => { setHist([]); localStorage.removeItem('mathlab.history') }}>Limpar</button>}</div>}
      {tab === 'Configurações' && <div className={card}><label className="text-sm"><input type="checkbox" checked={dark} onChange={() => setDark(!dark)} /> Tema escuro</label></div>}
    </div>
  )
}
