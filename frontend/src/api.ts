async function call(path: string, body?: unknown) {
  const r = await fetch('/api' + path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const j = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : `Erro ${r.status}`)
  return j
}
export type Step = { title: string; latex: string }
export type EvalResult = { input: string; result: string; latex: string; exact: boolean; steps: Step[]; verified: boolean | null; note: string | null }
export type Param = { name: string; required: boolean; default: unknown }
export type Method = { name: string; doc: string; params: Param[] }
export const api = {
  evaluate: (expression: string): Promise<EvalResult> => call('/evaluate', { expression }),
  modules: (): Promise<Record<string, Method[]>> => call('/modules'),
  run: (module: string, method: string, params: object) => call(`/run/${module}`, { method, params }),
  plot: (body: object) => call('/plot', body),
}
