# MathLab

## Backend (FastAPI + SymPy + NumPy)
    cd backend && pip install -r requirements.txt && pytest && uvicorn mathlab.api:app --reload
Endpoints: `POST /api/evaluate`, `GET /api/modules`, `POST /api/run/{physics|programming|numeric}`, `POST /api/plot`.
Novas ferramentas: adicione a função ao `METHODS` do módulo; ela aparece sozinha na interface.

## Frontend (React + TS + Vite + Tailwind + KaTeX + Plotly)
    cd frontend && npm install && npm run dev      # http://localhost:5173 (proxy /api → :8000)
