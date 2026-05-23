"""Multi-machine HTTP server (SPEC-007).

The 2Key server-API: every Calyie endpoint (Razer Blade, Qosimo, Phone, etc.)
talks to this FastAPI app instead of running STT/LLM/TTS locally. Pipelines
from SPEC-002-005 are reused 1:1 — the server is a thin HTTP wrapper, not a
re-implementation. Construction is lazy: importing the module does not build
any pipeline or open a network socket.
"""
