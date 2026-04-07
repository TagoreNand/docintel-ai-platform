import { useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000/api/v1";

type DocumentRecord = {
  id: string;
  filename: string;
  status: string;
  source: string;
  doc_type?: string | null;
  confidence?: number | null;
  anomaly_score?: number | null;
  summary?: string | null;
  created_at: string;
};

type ReviewTask = {
  id: string;
  document_id: string;
  reason: string;
  status: string;
  priority: string;
};

type Evidence = {
  document_id: string;
  filename: string;
  score: number;
  text: string;
};

type SearchResponse = {
  question: string;
  answer: string;
  evidence: Evidence[];
  related_entities: Record<string, string[]>;
};

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [reviewTasks, setReviewTasks] = useState<ReviewTask[]>([]);
  const [analytics, setAnalytics] = useState<Record<string, unknown> | null>(null);
  const [question, setQuestion] = useState("Which invoice mentions Nova Industrial Supplies?");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [uploading, setUploading] = useState(false);

  const refresh = async () => {
    const [docs, tasks, analyticsResponse] = await Promise.all([
      fetch(`${API_BASE}/documents`).then((res) => res.json()),
      fetch(`${API_BASE}/review/tasks`).then((res) => res.json()),
      fetch(`${API_BASE}/analytics/overview`).then((res) => res.json())
    ]);
    setDocuments(docs);
    setReviewTasks(tasks);
    setAnalytics(analyticsResponse);
  };

  useEffect(() => {
    refresh().catch(console.error);
  }, []);

  const upload = async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData
    });
    await refresh();
    setUploading(false);
  };

  const ingestSamples = async () => {
    await fetch(`${API_BASE}/documents/ingest-sample`, { method: "POST" });
    await refresh();
  };

  const runSearch = async () => {
    const payload = await fetch(`${API_BASE}/search/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 })
    }).then((res) => res.json());
    setSearchResult(payload);
  };

  const approveTask = async (taskId: string) => {
    await fetch(`${API_BASE}/review/tasks/${taskId}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome: "approved", notes: "Approved from dashboard." })
    });
    await refresh();
  };

  const metrics = useMemo(() => analytics ?? {}, [analytics]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>DocIntel AI</h1>
        <p className="muted">Senior-level enterprise document intelligence platform</p>
        <button className="primary" onClick={ingestSamples}>Ingest Sample Data</button>
        <label className="upload">
          <span>{uploading ? "Uploading..." : "Upload Document"}</span>
          <input
            type="file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) upload(file).catch(console.error);
            }}
          />
        </label>
      </aside>

      <main className="content">
        <section className="hero">
          <div>
            <h2>Operations Console</h2>
            <p>Upload, search, review, and analyze extracted intelligence from enterprise documents.</p>
          </div>
          <div className="stats-grid">
            <MetricCard label="Documents" value={String(metrics.documents_total ?? 0)} />
            <MetricCard label="Review Tasks" value={String(metrics.review_tasks_total ?? 0)} />
            <MetricCard label="Auto Approved" value={String(metrics.documents_auto_approved ?? 0)} />
            <MetricCard label="Avg Anomaly" value={String(metrics.average_anomaly_score ?? 0)} />
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>Grounded Search</h3>
          </div>
          <div className="search-row">
            <input value={question} onChange={(e) => setQuestion(e.target.value)} />
            <button className="primary" onClick={runSearch}>Search</button>
          </div>
          {searchResult && (
            <div className="search-results">
              <p><strong>Answer:</strong> {searchResult.answer}</p>
              <div className="evidence-list">
                {searchResult.evidence.map((item) => (
                  <div className="evidence-card" key={`${item.document_id}-${item.score}`}>
                    <h4>{item.filename}</h4>
                    <p className="muted">Score: {item.score}</p>
                    <p>{item.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="two-col">
          <div className="panel">
            <div className="panel-header">
              <h3>Documents</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Confidence</th>
                  <th>Anomaly</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div>{doc.filename}</div>
                      <div className="muted">{doc.summary}</div>
                    </td>
                    <td>{doc.doc_type ?? "unknown"}</td>
                    <td>{doc.status}</td>
                    <td>{typeof doc.confidence === "number" ? doc.confidence.toFixed(2) : "-"}</td>
                    <td>{typeof doc.anomaly_score === "number" ? doc.anomaly_score.toFixed(2) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3>Review Queue</h3>
            </div>
            {reviewTasks.length === 0 ? (
              <p className="muted">No review tasks yet.</p>
            ) : (
              reviewTasks.map((task) => (
                <div className="review-card" key={task.id}>
                  <div>
                    <strong>{task.priority.toUpperCase()}</strong>
                    <p>{task.reason}</p>
                    <p className="muted">Document ID: {task.document_id}</p>
                  </div>
                  <button className="secondary" onClick={() => approveTask(task.id)}>
                    Approve
                  </button>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="muted">{label}</div>
    </div>
  );
}
