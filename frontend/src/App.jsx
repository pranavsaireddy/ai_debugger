/**
 * dashboard.jsx — AI System Debugger Dashboard
 *
 * A production-grade React dashboard that connects to the FastAPI backend
 * and provides real-time visibility into every stage of the RAG pipeline.
 *
 * Panels:
 *   1. Query List        — run queries, see status/latency/issue
 *   2. Trace View        — per-query pipeline breakdown
 *   3. Issue Panel       — root cause + suggestions
 *   4. Metrics Dashboard — charts (latency, error rate, retrieval scores)
 */

import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

// ── Config ─────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

// ── Colour palette (dark terminal aesthetic) ────────────────────────────────
const C = {
  bg:      "#0a0e17",
  panel:   "#111827",
  border:  "#1f2937",
  accent:  "#3b82f6",
  green:   "#22c55e",
  yellow:  "#f59e0b",
  red:     "#ef4444",
  muted:   "#6b7280",
  text:    "#e2e8f0",
  dim:     "#94a3b8",
  PASS:    "#22c55e",
  FAIL:    "#ef4444",
};

const PIE_COLORS = ["#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#22c55e"];

// ── Sample documents for demo ingestion ────────────────────────────────────
const SAMPLE_DOCS = [
  {
    title: "Introduction to RAG",
    content:
      "Retrieval-Augmented Generation (RAG) is a technique that enhances large language " +
      "models by retrieving relevant documents from a knowledge base before generating a response. " +
      "This grounds the model's output in factual, up-to-date information and reduces hallucination. " +
      "RAG systems typically consist of a document ingestion pipeline, a vector store, a retriever, " +
      "and a generator (LLM). The retriever finds top-K relevant chunks using cosine similarity " +
      "between query and document embeddings. These chunks are injected into the prompt as context.",
  },
  {
    title: "Hallucination in LLMs",
    content:
      "Hallucination occurs when a language model generates confident but factually incorrect output. " +
      "This can happen when the model's parametric knowledge conflicts with retrieved context, " +
      "or when the prompt lacks grounding constraints. Common mitigations include lowering temperature, " +
      "adding explicit system instructions ('do not use prior knowledge'), and post-generation " +
      "semantic similarity checks between response and context. Sentence-level grounding analysis " +
      "measures what fraction of response sentences have high cosine similarity to retrieved passages.",
  },
  {
    title: "Vector Embeddings",
    content:
      "Embeddings are dense vector representations of text in a high-dimensional space where " +
      "semantically similar texts are geometrically close. OpenAI's text-embedding-ada-002 model " +
      "produces 1536-dimensional vectors. Cosine similarity (dot product of L2-normalised vectors) " +
      "measures semantic overlap between 0 (unrelated) and 1 (identical). FAISS (Facebook AI " +
      "Similarity Search) is an efficient library for nearest-neighbour search over millions of vectors.",
  },
];

const SAMPLE_QUERIES = [
  "What is Retrieval-Augmented Generation?",
  "How does hallucination happen in language models?",
  "Explain cosine similarity for embeddings",
  "What is quantum computing?",   // off-topic → triggers retrieval failure
  "Tell me about dinosaurs",       // off-topic → triggers hallucination risk
];


// ── Utilities ───────────────────────────────────────────────────────────────
const fmt = (n, d = 3) => typeof n === "number" ? n.toFixed(d) : "—";
const pct = (n) => typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "—";
const ms = (n) => typeof n === "number" ? `${(n * 1000).toFixed(0)} ms` : "—";

function Badge({ label, color }) {
  return (
    <span style={{
      background: color + "22", color, border: `1px solid ${color}44`,
      borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700,
      letterSpacing: 0.5, textTransform: "uppercase",
    }}>
      {label}
    </span>
  );
}

function IssueColor(type) {
  const map = {
    NONE: C.green, RETRIEVAL_FAILURE: C.yellow,
    PROMPT_QUALITY: C.accent, HALLUCINATION: C.red, HIGH_LATENCY: C.muted,
  };
  return map[type] || C.dim;
}

function ScoreBar({ value, max = 1, warn = 0.5, danger = 0.3 }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  const color = value < danger ? C.red : value < warn ? C.yellow : C.green;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
      <div style={{ flex: 1, background: C.border, borderRadius: 3, height: 6 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 3, height: 6, transition: "width 0.4s" }} />
      </div>
      <span style={{ color, fontSize: 12, fontFamily: "monospace", minWidth: 42, textAlign: "right" }}>
        {fmt(value)}
      </span>
    </div>
  );
}


// ── Panel component ─────────────────────────────────────────────────────────
function Panel({ title, children, style = {} }) {
  return (
    <div style={{
      background: C.panel, border: `1px solid ${C.border}`,
      borderRadius: 8, padding: 20, ...style,
    }}>
      {title && (
        <div style={{
          fontSize: 11, fontWeight: 700, letterSpacing: 1.5,
          color: C.dim, textTransform: "uppercase", marginBottom: 14,
          borderBottom: `1px solid ${C.border}`, paddingBottom: 10,
        }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}


// ── Stage node for trace view ────────────────────────────────────────────────
function StageNode({ label, status, children }) {
  const color = status === "ok" ? C.green : status === "warn" ? C.yellow : C.red;
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 16 }}>
      <div style={{
        width: 10, height: 10, borderRadius: "50%", background: color,
        marginTop: 4, flexShrink: 0, boxShadow: `0 0 8px ${color}`,
      }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color, marginBottom: 6 }}>{label}</div>
        <div style={{ background: C.bg, borderRadius: 6, padding: 12, fontSize: 12, color: C.dim, lineHeight: 1.6 }}>
          {children}
        </div>
      </div>
    </div>
  );
}


// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("queries");
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [selectedLog, setSelectedLog] = useState(null);
  const [queryInput, setQueryInput] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [ingested, setIngested] = useState(false);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(null);

  // Check backend connection
  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.ok ? setConnected(true) : setConnected(false))
      .catch(() => setConnected(false));
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/logs`);
      const d = await r.json();
      setLogs(d.logs || []);
    } catch { /* offline demo */ }
  }, []);

  const refreshMetrics = useCallback(async () => {
    try {
      const r = await fetch(`${API}/metrics`);
      setMetrics(await r.json());
    } catch { /* offline demo */ }
  }, []);

  useEffect(() => {
    refreshLogs();
    refreshMetrics();
  }, []);

  const handleIngest = async () => {
    setIngestLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: SAMPLE_DOCS }),
      });
      if (r.ok) setIngested(true);
      else setError("Ingestion failed — check the backend.");
    } catch {
      setError("Cannot reach backend. Start FastAPI on port 8000.");
    }
    setIngestLoading(false);
  };

  const handleQuery = async (q = queryInput) => {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q.trim(), top_k: topK }),
      });
      const data = await r.json();
      setSelectedLog(data);
      await refreshLogs();
      await refreshMetrics();
      setTab("trace");
    } catch {
      setError("Query failed — ensure backend is running.");
    }
    setLoading(false);
    setQueryInput("");
  };

  // ── Tabs ─────────────────────────────────────────────────────────────────
  const TABS = [
    { id: "queries", label: "Query List" },
    { id: "trace", label: "Trace View" },
    { id: "issues", label: "Issues & Fixes" },
    { id: "metrics", label: "Metrics" },
  ];

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
      fontSize: 13,
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 28px", borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", gap: 20,
        background: "#0d1220",
      }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.accent, letterSpacing: 1 }}>
            ◈ AI SYSTEM DEBUGGER
          </span>
          <span style={{ fontSize: 10, color: C.muted, letterSpacing: 2 }}>
            RAG PIPELINE OBSERVABILITY v1.0
          </span>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 11, color: connected === null ? C.dim : connected ? C.green : C.red,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%",
              background: connected === null ? C.dim : connected ? C.green : C.red,
              display: "inline-block",
            }} />
            {connected === null ? "CONNECTING" : connected ? "BACKEND ONLINE" : "BACKEND OFFLINE"}
          </div>
          {!ingested && (
            <button
              onClick={handleIngest}
              disabled={ingestLoading}
              style={{
                background: C.accent + "22", border: `1px solid ${C.accent}`,
                color: C.accent, borderRadius: 4, padding: "6px 14px",
                cursor: "pointer", fontSize: 11, fontWeight: 700,
              }}
            >
              {ingestLoading ? "INGESTING…" : "⬆ LOAD SAMPLE DOCS"}
            </button>
          )}
          {ingested && (
            <Badge label="✓ DOCS LOADED" color={C.green} />
          )}
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        display: "flex", gap: 0, borderBottom: `1px solid ${C.border}`,
        padding: "0 28px", background: "#0d1220",
      }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "12px 20px", fontSize: 11, fontWeight: 700,
            letterSpacing: 1, color: tab === t.id ? C.accent : C.muted,
            borderBottom: `2px solid ${tab === t.id ? C.accent : "transparent"}`,
            transition: "color 0.2s",
          }}>
            {t.label.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Error bar */}
      {error && (
        <div style={{
          background: C.red + "22", borderBottom: `1px solid ${C.red}`,
          color: C.red, padding: "10px 28px", fontSize: 12,
        }}>
          ✕ {error}
        </div>
      )}

      {/* Query input */}
      <div style={{
        padding: "16px 28px", borderBottom: `1px solid ${C.border}`,
        display: "flex", gap: 10, alignItems: "center",
      }}>
        <input
          value={queryInput}
          onChange={e => setQueryInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleQuery()}
          placeholder="Enter query to debug… (press Enter)"
          style={{
            flex: 1, background: C.panel, border: `1px solid ${C.border}`,
            borderRadius: 4, padding: "10px 14px", color: C.text,
            fontFamily: "inherit", fontSize: 13, outline: "none",
          }}
        />
        <div style={{ display: "flex", gap: 6, alignItems: "center", color: C.dim, fontSize: 11 }}>
          top_k
          <input
            type="number" min={1} max={20} value={topK}
            onChange={e => setTopK(+e.target.value)}
            style={{
              width: 44, background: C.panel, border: `1px solid ${C.border}`,
              color: C.text, borderRadius: 4, padding: "8px 6px",
              fontFamily: "inherit", fontSize: 13, textAlign: "center",
            }}
          />
        </div>
        <button
          onClick={() => handleQuery()}
          disabled={loading || !queryInput.trim()}
          style={{
            background: loading ? C.muted : C.accent,
            border: "none", borderRadius: 4, padding: "10px 20px",
            color: "#fff", fontWeight: 700, fontSize: 12,
            cursor: loading ? "not-allowed" : "pointer", letterSpacing: 0.5,
          }}
        >
          {loading ? "RUNNING…" : "▶ DEBUG"}
        </button>
        {/* Quick samples */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {SAMPLE_QUERIES.map(q => (
            <button key={q} onClick={() => handleQuery(q)} style={{
              background: "none", border: `1px solid ${C.border}`,
              borderRadius: 4, padding: "6px 10px", color: C.dim,
              cursor: "pointer", fontSize: 10, maxWidth: 140,
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
            }} title={q}>
              {q.slice(0, 28)}…
            </button>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div style={{ padding: "20px 28px" }}>

        {/* ── TAB: Query List ─────────────────────────────────────────────── */}
        {tab === "queries" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <span style={{ color: C.dim, fontSize: 12 }}>
                {logs.length} queries logged
              </span>
              <button onClick={async () => {
                await fetch(`${API}/logs`, { method: "DELETE" });
                setLogs([]); setMetrics(null);
              }} style={{
                background: "none", border: `1px solid ${C.border}`,
                color: C.muted, borderRadius: 4, padding: "5px 12px",
                cursor: "pointer", fontSize: 11,
              }}>
                CLEAR LOGS
              </button>
            </div>

            {logs.length === 0 && (
              <Panel>
                <div style={{ textAlign: "center", color: C.muted, padding: "40px 0" }}>
                  No queries yet. Load sample docs and run a query above.
                </div>
              </Panel>
            )}

            {logs.map(log => (
              <div
                key={log.id}
                onClick={() => { setSelectedLog(log); setTab("trace"); }}
                style={{
                  background: C.panel, border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: "14px 18px", marginBottom: 8,
                  cursor: "pointer", display: "grid",
                  gridTemplateColumns: "1fr auto auto auto auto",
                  gap: 16, alignItems: "center",
                  transition: "border-color 0.2s",
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = C.accent + "66"}
                onMouseLeave={e => e.currentTarget.style.borderColor = C.border}
              >
                <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: C.text }}>
                  {log.query}
                </div>
                <Badge
                  label={log.status?.toUpperCase()}
                  color={log.status === "pass" ? C.green : C.red}
                />
                <span style={{ color: C.dim, fontSize: 11 }}>{ms(log.latency)}</span>
                <Badge
                  label={log.primary_issue?.replace(/_/g, " ")}
                  color={IssueColor(log.primary_issue)}
                />
                <span style={{ color: C.muted, fontSize: 10 }}>
                  {log.timestamp?.slice(11, 19)}
                </span>
              </div>
            ))}
          </div>
        )}


        {/* ── TAB: Trace View ─────────────────────────────────────────────── */}
        {tab === "trace" && selectedLog && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>
            {/* Pipeline trace */}
            <div>
              <Panel title="Pipeline Trace" style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 12, color: C.dim, marginBottom: 16, wordBreak: "break-word" }}>
                  <strong style={{ color: C.text }}>Query:</strong> {selectedLog.query}
                </div>

                {/* Stage: Retrieval */}
                <StageNode
                  label="RETRIEVAL"
                  status={selectedLog.evaluations?.retrieval?.retrieval_ok ? "ok" : "warn"}
                >
                  <div>Docs retrieved: <strong>{selectedLog.retrieved_docs?.length ?? 0}</strong></div>
                  <div>Avg similarity: <strong style={{ color: selectedLog.evaluations?.retrieval?.retrieval_ok ? C.green : C.yellow }}>
                    {fmt(selectedLog.evaluations?.retrieval?.avg_score)}
                  </strong></div>
                  <div>Status: <strong>{selectedLog.evaluations?.retrieval?.flag}</strong></div>
                  {selectedLog.retrieved_docs?.slice(0, 2).map((d, i) => (
                    <div key={i} style={{
                      marginTop: 8, background: "#1a2235", borderRadius: 4,
                      padding: "8px 10px", fontSize: 11,
                    }}>
                      <div style={{ color: C.accent, marginBottom: 4 }}>
                        [{d.metadata?.source}] score={fmt(d.score)}
                      </div>
                      <div style={{ color: C.dim, lineHeight: 1.5 }}>
                        {d.text?.slice(0, 180)}…
                      </div>
                    </div>
                  ))}
                </StageNode>

                {/* Arrow */}
                <div style={{ color: C.border, marginLeft: 4, marginBottom: 8 }}>│</div>

                {/* Stage: Prompt */}
                <StageNode
                  label="PROMPT CONSTRUCTION"
                  status={selectedLog.evaluations?.prompt?.score >= 70 ? "ok" : "warn"}
                >
                  <div>Score: <strong>{selectedLog.evaluations?.prompt?.score}/100</strong></div>
                  <div>Token estimate: <strong>{selectedLog.evaluations?.prompt?.token_estimate}</strong></div>
                  <div>Status: <strong>{selectedLog.evaluations?.prompt?.flag}</strong></div>
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ cursor: "pointer", color: C.accent, fontSize: 11 }}>View prompt snippet</summary>
                    <pre style={{
                      marginTop: 6, padding: 10, background: "#1a2235",
                      borderRadius: 4, fontSize: 10, color: C.dim,
                      overflow: "auto", maxHeight: 200, whiteSpace: "pre-wrap",
                    }}>
                      {selectedLog.prompt?.slice(0, 600)}…
                    </pre>
                  </details>
                </StageNode>

                <div style={{ color: C.border, marginLeft: 4, marginBottom: 8 }}>│</div>

                {/* Stage: Generation */}
                <StageNode
                  label="LLM GENERATION"
                  status={selectedLog.evaluations?.hallucination?.hallucination_risk ? "error" : "ok"}
                >
                  <div>Latency: <strong>{ms(selectedLog.latency)}</strong></div>
                  <div>Context similarity: <strong style={{
                    color: (selectedLog.evaluations?.hallucination?.response_context_similarity ?? 0) < 0.6 ? C.red : C.green,
                  }}>
                    {fmt(selectedLog.evaluations?.hallucination?.response_context_similarity)}
                  </strong></div>
                  <div>Grounded sentences: <strong>
                    {pct(selectedLog.evaluations?.hallucination?.grounded_sentence_ratio)}
                  </strong></div>
                  <div>Hallucination risk: <strong style={{
                    color: selectedLog.evaluations?.hallucination?.hallucination_risk ? C.red : C.green,
                  }}>
                    {selectedLog.evaluations?.hallucination?.hallucination_risk ? "YES ⚠" : "NO ✓"}
                  </strong></div>
                  <div style={{
                    marginTop: 10, padding: "10px 12px", background: "#1a2235",
                    borderRadius: 4, color: C.text, lineHeight: 1.6, fontSize: 12,
                  }}>
                    <strong style={{ color: C.dim }}>Response:</strong><br />
                    {selectedLog.response}
                  </div>
                </StageNode>
              </Panel>
            </div>

            {/* Metrics sidebar */}
            <div>
              <Panel title="Stage Metrics" style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Retrieval Score</div>
                    <ScoreBar value={selectedLog.evaluations?.retrieval?.avg_score ?? 0} />
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Prompt Quality</div>
                    <ScoreBar value={(selectedLog.evaluations?.prompt?.score ?? 0) / 100} warn={0.7} danger={0.4} />
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Context Similarity</div>
                    <ScoreBar value={selectedLog.evaluations?.hallucination?.response_context_similarity ?? 0} warn={0.6} danger={0.4} />
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Grounded Sentences</div>
                    <ScoreBar value={selectedLog.evaluations?.hallucination?.grounded_sentence_ratio ?? 0} warn={0.6} danger={0.3} />
                  </div>
                </div>
              </Panel>

              <Panel title="Overall Status">
                <div style={{ textAlign: "center", padding: "10px 0" }}>
                  <div style={{
                    fontSize: 32,
                    color: selectedLog.status === "pass" ? C.green : C.red,
                  }}>
                    {selectedLog.status === "pass" ? "✓" : "✕"}
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, marginTop: 4, color: selectedLog.status === "pass" ? C.green : C.red }}>
                    {selectedLog.status?.toUpperCase()}
                  </div>
                  <div style={{ fontSize: 11, color: C.dim, marginTop: 4 }}>
                    {selectedLog.issue_count} issue(s) detected
                  </div>
                </div>
              </Panel>
            </div>
          </div>
        )}

        {tab === "trace" && !selectedLog && (
          <Panel>
            <div style={{ textAlign: "center", color: C.muted, padding: "40px 0" }}>
              Run a query to see the pipeline trace.
            </div>
          </Panel>
        )}


        {/* ── TAB: Issues & Fixes ─────────────────────────────────────────── */}
        {tab === "issues" && selectedLog && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {/* Issues */}
            <Panel title="Detected Issues">
              {selectedLog.analysis?.issues?.length === 0 && (
                <div style={{ color: C.green, textAlign: "center", padding: "20px 0" }}>
                  ✓ No issues detected
                </div>
              )}
              {selectedLog.analysis?.issues?.map((issue, i) => (
                <div key={i} style={{
                  background: C.bg, border: `1px solid ${IssueColor(issue.issue_type)}44`,
                  borderLeft: `3px solid ${IssueColor(issue.issue_type)}`,
                  borderRadius: 4, padding: "12px 14px", marginBottom: 10,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <Badge label={issue.issue_type?.replace(/_/g, " ")} color={IssueColor(issue.issue_type)} />
                    <span style={{ fontSize: 11, color: C.dim }}>
                      confidence: {pct(issue.confidence)}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6 }}>{issue.description}</div>
                  <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
                    Stage: <strong style={{ color: C.text }}>{issue.affected_stage}</strong>
                  </div>
                </div>
              ))}
            </Panel>

            {/* Suggestions */}
            <Panel title="Suggested Fixes">
              {selectedLog.suggestions?.map((s, i) => (
                <div key={i} style={{
                  background: C.bg, border: `1px solid ${C.border}`,
                  borderRadius: 4, padding: "12px 14px", marginBottom: 10,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <strong style={{ color: C.text, fontSize: 12 }}>{s.action}</strong>
                    <Badge
                      label={s.priority}
                      color={s.priority === "HIGH" ? C.red : s.priority === "MEDIUM" ? C.yellow : C.muted}
                    />
                  </div>
                  <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 8 }}>
                    {s.detail}
                  </div>
                  {s.code_hint && (
                    <pre style={{
                      background: "#1a2235", borderRadius: 4, padding: "8px 10px",
                      fontSize: 11, color: C.accent, overflow: "auto",
                    }}>
                      {s.code_hint}
                    </pre>
                  )}
                </div>
              ))}
            </Panel>
          </div>
        )}

        {tab === "issues" && !selectedLog && (
          <Panel>
            <div style={{ textAlign: "center", color: C.muted, padding: "40px 0" }}>
              Run a query first, then return here to see issues and fixes.
            </div>
          </Panel>
        )}


        {/* ── TAB: Metrics ─────────────────────────────────────────────────── */}
        {tab === "metrics" && (
          <div>
            {/* KPI cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 20 }}>
              {[
                { label: "Total Queries", value: metrics?.total ?? 0, color: C.accent },
                { label: "Error Rate", value: pct(metrics?.error_rate), color: C.red },
                { label: "Hallucination Rate", value: pct(metrics?.hallucination_rate), color: C.yellow },
                { label: "Avg Latency", value: ms(metrics?.avg_latency), color: C.green },
              ].map(k => (
                <Panel key={k.label}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 28, fontWeight: 700, color: k.color }}>{k.value}</div>
                    <div style={{ fontSize: 11, color: C.dim, marginTop: 4 }}>{k.label}</div>
                  </div>
                </Panel>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
              {/* Latency over time */}
              <Panel title="Latency Over Time (seconds)">
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={metrics?.latency_series ?? []}>
                    <CartesianGrid stroke={C.border} />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis stroke={C.dim} fontSize={10} />
                    <Tooltip
                      contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, fontSize: 11 }}
                      formatter={v => [`${(v * 1000).toFixed(0)} ms`, "Latency"]}
                    />
                    <Line type="monotone" dataKey="latency" stroke={C.accent} dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>

              {/* Issue breakdown */}
              <Panel title="Issue Breakdown">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={Object.entries(metrics?.issue_breakdown ?? {}).map(([k, v]) => ({ name: k.replace(/_/g, " "), value: v }))}
                      cx="50%" cy="50%" outerRadius={70}
                      dataKey="value" label={({ name, value }) => `${value}`}
                      labelLine={false}
                    >
                      {Object.keys(metrics?.issue_breakdown ?? {}).map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, fontSize: 11 }} />
                  </PieChart>
                </ResponsiveContainer>
              </Panel>

              {/* Retrieval score trend */}
              <Panel title="Avg Retrieval Score">
                <div style={{ padding: "20px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 48, fontWeight: 700, color: (metrics?.avg_retrieval_score ?? 0) < 0.5 ? C.red : C.green }}>
                    {fmt(metrics?.avg_retrieval_score, 2)}
                  </div>
                  <div style={{ fontSize: 11, color: C.dim, marginTop: 4 }}>
                    Mean cosine similarity across all queries
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <ScoreBar value={metrics?.avg_retrieval_score ?? 0} />
                  </div>
                </div>
              </Panel>

              {/* Hallucination vs Pass */}
              <Panel title="Hallucination vs Pass">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={[
                    { name: "Pass", value: (metrics?.total ?? 0) - (metrics?.hallucination_count ?? 0), fill: C.green },
                    { name: "Hallucination", value: metrics?.hallucination_count ?? 0, fill: C.red },
                  ]}>
                    <CartesianGrid stroke={C.border} />
                    <XAxis dataKey="name" stroke={C.dim} fontSize={10} />
                    <YAxis stroke={C.dim} fontSize={10} allowDecimals={false} />
                    <Tooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, fontSize: 11 }} />
                    <Bar dataKey="value">
                      {[{ fill: C.green }, { fill: C.red }].map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}