#!/usr/bin/env python3
"""
GraphRAG Demo: Enhancing RAG with Knowledge-Graph Retrieval
============================================================
Compares three retrieval strategies over a corpus:

  1. vector  - vanilla RAG (embedding similarity over document chunks)
  2. graph   - GraphRAG (LLM-extracted knowledge graph + multi-hop traversal)
  3. hybrid  - graph facts + top vector chunk

Pipeline:
  docs -> LLM triple extraction -> NetworkX knowledge graph (kg.json)
  query -> seed entities -> k-hop subgraph -> facts + linked docs -> LLM answer

Default LLM: GLM-4.7-Flash via Z.ai (free tier). Any OpenAI-compatible
endpoint works (Z.ai, BigModel, OpenAI, Ollama, LM Studio, ...).

SETUP
-----
  pip install openai sentence-transformers networkx numpy pyvis matplotlib
  # optional, only if you ingest such files:
  pip install pypdf python-docx

  Get an API key from your Z.ai / BigModel console, then:
    Linux/Mac : export ZAI_API_KEY="your-key"
    Windows   : set ZAI_API_KEY=your-key
  (OpenAI also works: just set OPENAI_API_KEY instead - model auto-switches.)

  Useful env overrides:
    LLM_MODEL=glm-4.5-flash                    # if "glm-4.7-flash" is rejected
    LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/   # mainland-China accounts
    VECTOR_K=3                                 # fairer baseline for custom docs
    HF_ENDPOINT=https://hf-mirror.com          # if HuggingFace is unreachable

USAGE
-----
  python graphrag_demo.py                       # full run: eval + visualize
  python graphrag_demo.py build                 # (re)build the knowledge graph
  python graphrag_demo.py ingest report.pdf     # use YOUR document (.txt/.md/.pdf/.docx)
  python graphrag_demo.py ask "Who is the CEO of the company that acquired DataFlow Systems?"
  python graphrag_demo.py eval                  # benchmark only
  python graphrag_demo.py viz --query "..."     # graph HTML with query path highlighted
  python graphrag_demo.py reset                 # back to the built-in demo corpus

OUTPUT FILES: kg.json, llm_cache.json, docs.json, results.json/.csv/.png,
              knowledge_graph.html
Re-runs are instant and free: every LLM response is cached in llm_cache.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter

import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("graphrag")

# ---------------------------------------------------------------------------
# Configuration (everything overridable via environment variables)
# ---------------------------------------------------------------------------
ZAI_KEY = os.getenv("ZAI_API_KEY") or os.getenv("BIGMODEL_API_KEY")
OAI_KEY = os.getenv("OPENAI_API_KEY")
if ZAI_KEY:
    _KEY, _BASE, _MODEL = ZAI_KEY, "https://api.z.ai/api/paas/v4/", "glm-4.7-flash"
elif OAI_KEY:
    _KEY, _BASE, _MODEL = OAI_KEY, "https://api.openai.com/v1", "gpt-4o-mini"
else:
    _KEY, _BASE, _MODEL = None, "https://api.z.ai/api/paas/v4/", "glm-4.5-flash"

API_KEY = _KEY
BASE_URL = os.getenv("LLM_BASE_URL") or _BASE
LLM_MODEL = os.getenv("LLM_MODEL") or _MODEL
DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "1").lower() in ("1", "true", "yes")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

KG_PATH = "kg.json"
CACHE_PATH = "llm_cache.json"
DOCS_PATH = "docs.json"
QUESTIONS_PATH = "questions.json"
RESULTS_JSON, RESULTS_CSV, RESULTS_PNG = "results.json", "results.csv", "results.png"
VIZ_PATH = "knowledge_graph.html"

HOPS = int(os.getenv("HOPS", "2"))            # graph traversal depth per query
VECTOR_K = int(os.getenv("VECTOR_K", "1"))    # chunks fed to the vanilla baseline.
             # Demo corpus = 1 fact per chunk, so 1 is honest. For real ingested
             # documents use VECTOR_K=3 (see guide) for a fair comparison.
GRAPH_DOCS = int(os.getenv("GRAPH_DOCS", "3"))   # source docs attached to graph facts
MAX_FACTS = int(os.getenv("MAX_FACTS", "60"))    # cap out facts fed to the LLM
SEED_TOP_N = int(os.getenv("SEED_TOP_N", "4"))
SEED_SIM_THR = float(os.getenv("SEED_SIM_THR", "0.30"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))

GENERIC_NODES = {"company", "companies", "startup", "university", "team",
                 "product", "award", "investor", "engineering", "money",
                 "cloud computing", "computer science", "networking company",
                 "data analytics company", "agritech startup"}

# ---------------------------------------------------------------------------
# Built-in demo corpus: fictional startup ecosystem (ground truth fully known,
# so every benchmark number is reproducible). Ingested documents override it.
# ---------------------------------------------------------------------------
DEMO_DOCS = [
    "Alice Chen founded DataFlow Systems in 2018. DataFlow Systems is a data analytics company headquartered in Berlin.",
    "DataFlow Systems was acquired by CloudNine Technologies in 2023 for 50 million dollars.",
    "CloudNine Technologies is a cloud computing company based in Munich. Its CEO is Marcus Webb.",
    "Marcus Webb previously worked at Speedy Labs, where he led the infrastructure team.",
    "Speedy Labs was founded by Priya Sharma in 2015 in Bangalore.",
    "Priya Sharma studied computer science at the Indian Institute of Technology.",
    "DataFlow Systems' head of engineering is Tom Becker, who joined in 2019.",
    "Tom Becker previously worked at NetSphere, a networking company in Hamburg.",
    "NetSphere was founded by Hans Muller, who is also an angel investor.",
    "Hans Muller invested in GreenLeaf, an agritech startup in Stuttgart.",
    "GreenLeaf's CEO is Lena Fischer, who won the German Startup Award in 2022.",
    "CloudNine Technologies has acquired two companies so far: DataFlow Systems and NetSphere.",
    "NetSphere's main product is FastRoute, a router firmware.",
    "DataFlow Systems' flagship product is InsightBoard, a data visualization tool.",
    "Lena Fischer taught startup strategy at Stuttgart University before joining GreenLeaf.",
    "GreenLeaf's flagship product is CropSense, a soil-monitoring sensor.",
]

# Canonical names, so "DataFlow" and "DataFlow Systems" merge into ONE node.
ALIASES = {
    "dataflow": "DataFlow Systems",
    "cloudnine": "CloudNine Technologies",
    "netsphere": "NetSphere",
    "greenleaf": "GreenLeaf",
    "speedy labs": "Speedy Labs",
    "iit": "Indian Institute of Technology",
    "german startup award": "German Startup Award",
    "stuttgart university": "Stuttgart University",
    "webb": "Marcus Webb",
    "sharma": "Priya Sharma",
    "fischer": "Lena Fischer",
    "becker": "Tom Becker",
    "muller": "Hans Muller",
    "chen": "Alice Chen",
}

# (question, gold answer, category)  -- gold may be a str or a list
QUESTIONS = [
    ("What is the flagship product of DataFlow Systems?", "InsightBoard", "factoid"),
    ("Where is CloudNine Technologies headquartered?", "Munich", "factoid"),
    ("Who is the CEO of the company that acquired DataFlow Systems?", "Marcus Webb", "multi-hop"),
    ("Which university did the founder of Speedy Labs attend?", "Indian Institute of Technology", "multi-hop"),
    ("What company did the head of engineering of DataFlow Systems previously work at?", "NetSphere", "multi-hop"),
    ("Who invested in the startup whose CEO won the German Startup Award?", "Hans Muller", "multi-hop"),
    ("Which companies has CloudNine Technologies acquired?", ["DataFlow", "NetSphere"], "aggregation"),
    ("Who is the CFO of CloudNine Technologies?", "Unknown", "unanswerable"),
]

# ---------------------------------------------------------------------------
# Document loading / ingestion
# ---------------------------------------------------------------------------
def _doc_hash(docs: list) -> str:
    return hashlib.sha1("\n".join(docs).encode()).hexdigest()[:12]


def load_documents() -> list:
    """Ingested documents (docs.json) take precedence over the demo corpus."""
    if os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH) as f:
                docs = json.load(f)["docs"]
            if docs:
                log.info("using ingested corpus: %d chunks from %s", len(docs), DOCS_PATH)
                return docs
        except (json.JSONDecodeError, KeyError):
            log.warning("%s is corrupt - falling back to the demo corpus", DOCS_PATH)
    return DEMO_DOCS


DOCS = load_documents()


def read_file(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".csv", ".text"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit("PDF support requires:  pip install pypdf")
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            sys.exit("DOCX support requires:  pip install python-docx")
        return "\n\n".join(p.text for p in docx.Document(path).paragraphs)
    sys.exit(f"Unsupported type '{ext}' - use .txt, .md, .pdf or .docx")


def chunk_text(text: str) -> list:
    """Paragraph-aware chunker: short paragraphs pass through whole (keeps the
    one-fact-per-chunk property); long ones split at sentence boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paras:
        if len(para) <= CHUNK_MAX_CHARS:
            chunks.append(para)
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if cur and len(cur) + len(sent) + 1 > CHUNK_MAX_CHARS:
                chunks.append(cur.strip())
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            chunks.append(cur.strip())
    return chunks


def do_ingest(path: str):
    global DOCS, _doc_emb
    chunks = chunk_text(read_file(path))
    if len(chunks) < 3:
        sys.exit("Document too short - need at least a few paragraphs.")
    with open(DOCS_PATH, "w") as f:
        json.dump({"source": os.path.basename(path), "docs": chunks}, f, indent=1)
    DOCS = chunks
    _doc_emb = None                       # force re-embedding
    log.info("ingested '%s' -> %d chunks (avg %d chars)",
             path, len(chunks), sum(map(len, chunks)) // len(chunks))

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
G = nx.DiGraph()          # the knowledge graph (in-memory, persisted to kg.json)
entity_docs: dict = {}    # node -> [doc indices mentioning it]
node_emb: dict = {}       # node -> embedding (for seed matching)
_llm = None
_st_model = None
_doc_emb = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


class LLM:
    """Chat wrapper: disk-cached (identical prompts are free on re-run),
    retrying, and self-simplifying (drops the 'thinking' flag / temperature
    if the provider rejects them)."""

    def __init__(self):
        if not API_KEY:
            sys.exit("Error: no API key. Set ZAI_API_KEY (Z.ai/BigModel) "
                     "or OPENAI_API_KEY.")
        from openai import OpenAI
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)
        self.cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                self.cache = json.load(f)
        self._mode = 0   # 0 = temperature+thinking-flag, 1 = temperature, 2 = bare
        # fail fast with a clear message instead of 50 cryptic errors later
        probe = self.chat("Connection test. Reply with exactly: OK")
        if "ok" not in probe.lower():
            sys.exit("LLM smoke test failed - cannot reach the model.\n"
                     f"   model    : {LLM_MODEL}\n   base_url : {BASE_URL}\n"
                     "   fixes    : check the API key; verify the model id in your\n"
                     "              provider console (try: export LLM_MODEL=glm-4.5-flash);\n"
                     "              China mainland: export LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/")

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
        key = hashlib.sha1(f"{LLM_MODEL}|{system}|{temperature}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        msgs = ([] if system is None else [{"role": "system", "content": system}])
        msgs.append({"role": "user", "content": prompt})
        for attempt in range(5):
            mode = min(self._mode + attempt, 2)        # simplify on repeated failure
            kwargs = {"model": LLM_MODEL, "messages": msgs}
            if mode <= 1:
                kwargs["temperature"] = temperature
            if mode == 0 and DISABLE_THINKING:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            try:
                resp = self.client.chat.completions.create(**kwargs)
                out = (resp.choices[0].message.content or "").strip()
                if not out:
                    raise ValueError("empty response content")
                self._mode = mode                     # remember what worked
                self.cache[key] = out
                with open(CACHE_PATH, "w") as f:
                    json.dump(self.cache, f)
                return out
            except Exception as e:
                wait = min(2 ** attempt, 8)
                log.warning("LLM call failed (%s); retry %d/5 in %ds", e, attempt + 1, wait)
                time.sleep(wait)
        log.error("prompt failed after 5 attempts; returning empty string")
        return ""


def embed(texts):
    global _st_model
    if _st_model is None:
        log.info("loading embedding model %s ...", EMBED_MODEL)
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
    return np.asarray(_st_model.encode(list(texts), normalize_embeddings=True))

# ---------------------------------------------------------------------------
# Knowledge-graph construction
# ---------------------------------------------------------------------------
EXTRACT_SYS = "You extract structured knowledge-graph triples from text. Output JSON only."

EXTRACT_PROMPT = """Extract knowledge-graph triples from the text below.

Rules:
- Entities are specific proper nouns: people, organizations, products, places, awards.
  Never use generic nouns (company, team, award) and never dates, years, or money amounts.
- Use full canonical names (e.g. "DataFlow Systems", not "it" or "DataFlow").
- Relations are short lowercase verb phrases such as: founded, acquired, ceo_of,
  worked_at, studied_at, invested_in, headquartered_in, based_in, won, taught_at,
  head_of_engineering_of, product_of, led.

Text: \"\"\"{text}\"\"\"

Return ONLY a JSON array, e.g.:
[{{"head": "Alice Chen", "relation": "founded", "tail": "DataFlow Systems"}}]"""


def _parse_json_list(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found")
    return json.loads(text[start:end + 1])


def normalize_entity(raw: str) -> str:
    s = " ".join(raw.strip().strip(".").split())
    return ALIASES.get(s.lower(), s)


def extract_triples(llm: LLM, text: str) -> list:
    for i in range(2):
        suffix = ("" if i == 0 else
                  "\n\n(Your previous output was not valid JSON. "
                  "Output ONLY the JSON array, nothing else.)")
        raw = llm.chat(EXTRACT_PROMPT.format(text=text) + suffix, system=EXTRACT_SYS)
        try:
            data = _parse_json_list(raw)
            return [t for t in data
                    if isinstance(t, dict) and t.get("head") and t.get("tail")]
        except Exception:
            log.warning("could not parse triples JSON, retrying")
    return []


def build_graph(llm: LLM):
    G.clear()
    for i, doc in enumerate(DOCS, 1):
        log.info("extracting triples  %d/%d", i, len(DOCS))
        for t in extract_triples(llm, doc):
            h, tail = normalize_entity(t["head"]), normalize_entity(t["tail"])
            rel = str(t.get("relation", "related_to")).strip().lower().replace(" ", "_") or "related_to"
            if (not h or not tail or h == tail
                    or h.lower() in GENERIC_NODES or tail.lower() in GENERIC_NODES):
                continue
            if G.has_edge(h, tail):
                if rel not in G[h][tail]["rels"]:
                    G[h][tail]["rels"].append(rel)
            else:
                G.add_edge(h, tail, rels=[rel])
    with open(KG_PATH, "w") as f:
        json.dump({"doc_hash": _doc_hash(DOCS),
                   "edges": [[h, t, d["rels"]] for h, t, d in G.edges(data=True)]},
                  f, indent=1)
    log.info("knowledge graph: %d entities, %d relationships -> %s",
             G.number_of_nodes(), G.number_of_edges(), KG_PATH)
    hubs = sorted(G.degree, key=lambda x: -x[1])[:5]
    if hubs:
        log.info("hub entities: %s", ", ".join(f"{n} ({d})" for n, d in hubs))


def load_graph():
    G.clear()
    with open(KG_PATH) as f:
        data = json.load(f)
    for h, t, rels in data["edges"]:
        G.add_edge(h, t, rels=rels)
    log.info("loaded knowledge graph: %d entities, %d relationships",
             G.number_of_nodes(), G.number_of_edges())


def link_documents():
    """Map each entity to the documents that mention it (graph -> source text)."""
    global entity_docs
    entity_docs = {n: [] for n in G.nodes}
    for i, doc in enumerate(DOCS):
        dl = doc.lower()
        for n in G.nodes:
            if n.lower() in dl:
                entity_docs[n].append(i)


def _kg_is_stale() -> bool:
    if not os.path.exists(KG_PATH):
        return True
    try:
        with open(KG_PATH) as f:
            return json.load(f).get("doc_hash") != _doc_hash(DOCS)
    except json.JSONDecodeError:
        return True


def ensure_graph(rebuild: bool = False):
    if rebuild or _kg_is_stale():
        if not rebuild and os.path.exists(KG_PATH):
            log.info("documents changed since the graph was built - rebuilding")
        build_graph(get_llm())
    else:
        load_graph()
    link_documents()
    global node_emb
    names = list(G.nodes)
    node_emb = dict(zip(names, embed(names))) if names else {}

# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
def vector_context(query: str, k: int = VECTOR_K) -> list:
    global _doc_emb
    if _doc_emb is None:
        _doc_emb = embed(DOCS)
    sims = _doc_emb @ embed([query])[0]
    order = np.argsort(sims)[::-1][:k]
    return [f"[doc {i + 1}] {DOCS[i]}" for i in order]


def find_seeds(query: str) -> list:
    """Match query to graph entities: literal substring first, else embeddings."""
    q, qe = query.lower(), embed([query])[0]
    scored = []
    for n in G.nodes:
        base = n.lower()
        if base in GENERIC_NODES or len(base) < 4:
            continue
        s = 1.0 if base in q else float(np.dot(qe, node_emb[n]))
        scored.append((s, n))
    scored.sort(key=lambda x: -x[0])
    return [n for s, n in scored[:SEED_TOP_N] if s >= SEED_SIM_THR]


def expand_neighborhood(seeds: list, hops: int) -> set:
    """BFS over BOTH edge directions - the multi-hop traversal step."""
    seen, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(G.successors(n), G.predecessors(n))
        nxt -= seen
        seen.update(nxt)
        frontier = nxt
    return seen


def graph_context(query: str):
    """Returns (facts, doc_lines, seeds). No seeds -> global mode (whole graph)."""
    seeds = find_seeds(query)
    nodes = expand_neighborhood(seeds, HOPS) if seeds else set(G.nodes)
    facts = [f"- {h} -[{'/'.join(d['rels'])}]-> {t}"
             for h, t, d in G.edges(data=True)
             if h in nodes and t in nodes][:MAX_FACTS]
    counts = Counter()
    for n in nodes:
        counts.update(entity_docs.get(n, []))
    docs = [f"[doc {i + 1}] {DOCS[i]}" for i, _ in counts.most_common(GRAPH_DOCS)]
    return facts, docs, seeds

# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------
ANSWER_SYS = "You answer strictly and only from the provided context."


def answer(llm: LLM, question: str, context_lines: list) -> str:
    prompt = ("Context:\n" + "\n".join(context_lines) + f"\n\nQuestion: {question}\n\n"
              "Using ONLY the context above, answer concisely (ideally just the answer, "
              "a short phrase or a list). If the context does not contain the answer, "
              "reply exactly: Unknown")
    return llm.chat(prompt, system=ANSWER_SYS)


def run_method(llm: LLM, question: str, method: str):
    """Returns (answer, graph_facts_used)."""
    if method == "vector":
        facts, docs = [], vector_context(question)
    elif method == "graph":
        facts, docs, _ = graph_context(question)
    else:  # hybrid
        facts, docs, _ = graph_context(question)
        docs = docs + vector_context(question, k=1)
    lines = []
    if facts:
        lines.append("Knowledge-graph facts:")
        lines.extend(facts)
    if docs:
        lines.append("Source documents:")
        lines.extend(docs)
    return answer(llm, question, lines), facts

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_questions():
    if os.path.exists(QUESTIONS_PATH):
        with open(QUESTIONS_PATH) as f:
            qs = json.load(f)
        out = [(q["question"], q["gold"], q.get("category", "custom")) for q in qs]
        log.info("using custom question set: %s (%d questions)", QUESTIONS_PATH, len(out))
        return out
    return QUESTIONS


def score(answer_text: str, gold) -> bool:
    a = answer_text.lower()
    if isinstance(gold, list):
        return all(g.lower() in a for g in gold)
    if gold.lower() == "unknown":
        return any(k in a for k in ("unknown", "not specified", "not mentioned",
                                    "not stated", "not provided", "no information",
                                    "does not", "do not"))
    return gold.lower() in a


def run_eval():
    llm = get_llm()
    methods = ["vector", "graph", "hybrid"]
    results = []
    print("\n" + "=" * 74)
    print("BENCHMARK  (vector = vanilla RAG, graph = GraphRAG, hybrid = both)")
    print("=" * 74)
    for qi, (q, gold, cat) in enumerate(load_questions(), 1):
        row = {"n": qi, "category": cat, "question": q, "expected": gold}
        print(f"\n[{qi}] {cat} | {q}")
        print(f"     expected : {gold}")
        for m in methods:
            ans, _ = run_method(llm, q, m)
            ok = score(ans, gold)
            row[m] = {"answer": ans, "correct": ok}
            print(f"     {m:<8}: {'PASS' if ok else 'FAIL'}  {ans}")
        results.append(row)

    cats = []
    for r in results:
        if r["category"] not in cats:
            cats.append(r["category"])

    print("\n" + "-" * 74)
    print(f"{'method':<10}" + "".join(f"{c[:12]:>14}" for c in cats) + f"{'TOTAL':>10}")
    for m in methods:
        cells, total = [], 0
        for c in cats:
            rows = [r for r in results if r["category"] == c]
            k = sum(r[m]["correct"] for r in rows)
            total += k
            cells.append(f"{k}/{len(rows)}")
        print(f"{m:<10}" + "".join(f"{c:>14}" for c in cells) + f"{total}/{len(results):>8}")
    print("-" * 74)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "category", "question", "expected"] +
                   [x for m in methods for x in (f"{m}_answer", f"{m}_ok")])
        for r in results:
            w.writerow([r["n"], r["category"], r["question"], r["expected"]] +
                       [x for m in methods for x in (r[m]["answer"], r[m]["correct"])])
    make_chart(results, cats, methods)
    log.info("results saved: %s, %s, %s", RESULTS_JSON, RESULTS_CSV, RESULTS_PNG)


def make_chart(results, cats, methods):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed - skipping chart")
        return
    acc = {m: [np.mean([r[m]["correct"] for r in results if r["category"] == c])
               for c in cats] for m in methods}
    x, width = np.arange(len(cats)), 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1) * width, acc[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("GraphRAG vs vanilla vector RAG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_viz(query: str | None = None):
    if G.number_of_nodes() == 0:
        log.info("graph is empty - nothing to visualize")
        return
    try:
        from pyvis.network import Network
        try:
            net = Network(directed=True, height="700px", cdn_resources="in_line")
        except TypeError:                       # older pyvis
            net = Network(directed=True, height="700px")
    except ImportError:
        log.info("pyvis not installed - skipping visualization")
        return
    seeds, highlight = set(), set()
    if query:
        _, _, seeds = graph_context(query)
        highlight = expand_neighborhood(list(seeds), HOPS) if seeds else set()
    for n in G.nodes:
        color = "#e74c3c" if n in seeds else ("#f39c12" if n in highlight else "#97c2fc")
        docs = [i + 1 for i in entity_docs.get(n, [])]
        net.add_node(n, label=n, color=color, title=f"mentioned in doc(s) {docs}")
    for h, t, d in G.edges(data=True):
        net.add_edge(h, t, label=" / ".join(d["rels"]))
    net.save_graph(VIZ_PATH)
    log.info("graph visualization -> %s%s", VIZ_PATH,
             "  (red = query seeds, orange = traversed, blue = rest)" if query else "")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_ask(question: str, method: str | None = None):
    llm = get_llm()
    for m in ([method] if method else ["vector", "graph", "hybrid"]):
        ans, facts = run_method(llm, question, m)
        print(f"\n--- {m.upper()} ---\n{ans}")
        if facts:
            print("graph facts used:")
            print("\n".join(facts[:8]))


def main():
    p = argparse.ArgumentParser(description="GraphRAG demo: knowledge-graph enhanced RAG")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("run", help="eval + visualize (default)")
    sub.add_parser("build", help="(re)build the knowledge graph")
    pi = sub.add_parser("ingest", help="load a document (.txt/.md/.pdf/.docx) and build its graph")
    pi.add_argument("path")
    pa = sub.add_parser("ask", help="ask one question, compare methods")
    pa.add_argument("question")
    pa.add_argument("--method", choices=["vector", "graph", "hybrid"])
    sub.add_parser("eval", help="run the benchmark question set")
    pv = sub.add_parser("viz", help="export the knowledge graph as HTML")
    pv.add_argument("--query", help="highlight the subgraph used for this query")
    sub.add_parser("reset", help="return to the built-in demo corpus")
    args = p.parse_args()
    cmd = args.command or "run"

    if cmd == "build":
        ensure_graph(rebuild=True)
        make_viz()
        return
    if cmd == "ingest":
        do_ingest(args.path)
        ensure_graph(rebuild=True)
        make_viz()
        print('\nIngested. Tip: for a fair baseline on real documents use '
              '"VECTOR_K=3 python graphrag_demo.py eval".\n'
              'Try:  python graphrag_demo.py ask "<question about your document>"')
        return
    if cmd == "reset":
        for path in (DOCS_PATH, KG_PATH):
            if os.path.exists(path):
                os.remove(path)
        print("Cleared custom documents and graph. Next run uses the demo corpus.")
        return

    ensure_graph()
    if cmd == "ask":
        run_ask(args.question, args.method)
    elif cmd == "eval":
        run_eval()
    elif cmd == "viz":
        make_viz(args.query)
    else:  # run
        run_eval()
        make_viz()
        print("\nDone. Artifacts:\n"
              f"  {VIZ_PATH:<22} open in a browser (screenshot for slides)\n"
              f"  {RESULTS_PNG:<22} accuracy chart\n"
              f"  {RESULTS_JSON}/{RESULTS_CSV} full answers\n\n"
              "Live demo tip:\n"
              '  python graphrag_demo.py ask "Who is the CEO of the company that acquired DataFlow Systems?"')


if __name__ == "__main__":
    main()