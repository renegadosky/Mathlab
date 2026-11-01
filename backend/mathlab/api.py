from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .engine import evaluate, MathError
from . import registry, plotting

app = FastAPI(title="MathLab API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def _guard(fn, *args):
    try:
        return fn(*args)
    except MathError as e:
        raise HTTPException(status_code=422, detail=str(e))


class Req(BaseModel):
    expression: str


class RunReq(BaseModel):
    method: str
    params: dict = {}


class PlotReq(BaseModel):
    expressions: list[str]
    xmin: float
    xmax: float
    points: int = 600


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/evaluate")
def run_evaluate(req: Req):
    return _guard(evaluate, req.expression)


@app.get("/api/modules")
def modules():
    return registry.describe()


@app.post("/api/run/{module}")
def run_module(module: str, req: RunReq):
    if module not in registry.MODULES:
        raise HTTPException(status_code=404, detail="Módulo desconhecido.")
    return _guard(registry.MODULES[module].run, req.method, req.params)


@app.post("/api/plot")
def run_plot(req: PlotReq):
    return _guard(plotting.plot, req.expressions, req.xmin, req.xmax, req.points)
