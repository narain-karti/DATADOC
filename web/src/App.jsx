import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Activity, ArrowRight, Check, Code2, Database, Download, FileJson, MessageSquare, Play, RefreshCw, ShieldCheck, Terminal, X } from 'lucide-react';

const API_BASE = import.meta.env.VITE_DATADOC_API_URL || `${window.location.origin}/api`;
const SESSION_HEADERS = { 'X-DATADOC-SESSION': 'local' };

const defaultConfig = {
  target: '',
  task: 'auto',
  scaling: 'auto',
  drop_identifiers: false,
  clip_outliers: false,
};

function Metric({ label, value, tone = 'bg-white' }) {
  return (
    <div className={`border-2 border-black p-4 ${tone}`}>
      <div className="text-[10px] font-bold uppercase tracking-wider">{label}</div>
      <div className="mt-2 break-words text-2xl font-bold">{value ?? '—'}</div>
    </div>
  );
}

function Pill({ children, tone = 'bg-white' }) {
  return <span className={`inline-flex border-2 border-black px-2 py-1 text-[10px] font-bold uppercase ${tone}`}>{children}</span>;
}

function App() {
  const [view, setView] = useState('overview');
  const [metadata, setMetadata] = useState(null);
  const [profile, setProfile] = useState(null);
  const [plan, setPlan] = useState(null);
  const [fitState, setFitState] = useState(null);
  const [preview, setPreview] = useState(null);
  const [code, setCode] = useState('');
  const [config, setConfig] = useState(defaultConfig);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const requestConfig = useMemo(() => ({
    ...config,
    target: config.target.trim() || null,
  }), [config]);

  useEffect(() => {
    refreshProfile();
  }, []);

  async function refreshProfile() {
    setError('');
    try {
      const [metaResponse, profileResponse] = await Promise.all([
        axios.get(`${API_BASE}/dataset/metadata`, { headers: SESSION_HEADERS }),
        axios.get(`${API_BASE}/pipeline/profile`, { headers: SESSION_HEADERS }),
      ]);
      setMetadata(metaResponse.data);
      setProfile(profileResponse.data);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Unable to connect to the local DATADOC server.');
    }
  }

  async function createPlan() {
    setBusy(true);
    setError('');
    try {
      const response = await axios.post(`${API_BASE}/pipeline/plan`, requestConfig, { headers: SESSION_HEADERS });
      setPlan(response.data);
      setView('pipeline');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Could not create the transformation plan.');
    } finally {
      setBusy(false);
    }
  }

  async function fitPipeline() {
    setBusy(true);
    setError('');
    try {
      const response = await axios.post(`${API_BASE}/pipeline/fit`, requestConfig, { headers: SESSION_HEADERS });
      setFitState(response.data);
      setProfile(response.data.profile);
      setPlan(response.data.plan);
      const previewResponse = await axios.get(`${API_BASE}/pipeline/preview`, { headers: SESSION_HEADERS });
      setPreview(previewResponse.data);
      setView('pipeline');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Could not fit the pipeline.');
    } finally {
      setBusy(false);
    }
  }

  async function loadCode() {
    setBusy(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE}/pipeline/export/code`, { headers: SESSION_HEADERS });
      setCode(response.data);
      setView('code');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Fit a pipeline before exporting code.');
    } finally {
      setBusy(false);
    }
  }

  async function downloadCsv() {
    setError('');
    try {
      const response = await axios.get(`${API_BASE}/dataset/export/csv`, {
        headers: SESSION_HEADERS,
        responseType: 'blob',
      });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'transformed_data.csv';
      link.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Fit a pipeline before downloading data.');
    }
  }

  async function sendChatMessage(event) {
    event.preventDefault();
    if (!chatInput.trim()) return;
    const message = chatInput.trim();
    setChatInput('');
    setChatHistory((history) => [...history, { role: 'user', content: message }]);
    try {
      const response = await axios.post(`${API_BASE}/agent/chat`, { message }, { headers: SESSION_HEADERS });
      setChatHistory((history) => [...history, { role: 'assistant', content: response.data.response }]);
    } catch (requestError) {
      setChatHistory((history) => [...history, { role: 'assistant', content: requestError.response?.data?.detail || 'AI is unavailable. The deterministic pipeline still works offline.' }]);
    }
  }

  const roles = profile?.roles || [];
  const findings = profile?.findings || [];
  const operations = plan?.operations || [];
  const outputColumns = fitState ? Object.keys(fitState.output_schema || {}) : [];

  return (
    <div className="flex min-h-screen w-full bg-bg-workspace font-mono text-black">
      <aside className="hidden w-[112px] shrink-0 flex-col border-r-2 border-black bg-bg-sidebar py-6 md:flex">
        <div className="mx-auto flex h-12 w-12 items-center justify-center bg-black text-xl font-bold text-white">DD</div>
        <nav className="mt-12 flex flex-col gap-2">
          <NavButton active={view === 'overview'} icon={<Database size={19} />} label="Profile" onClick={() => setView('overview')} />
          <NavButton active={view === 'pipeline'} icon={<Play size={19} />} label="Pipeline" onClick={() => setView('pipeline')} />
          <NavButton active={view === 'code'} icon={<Code2 size={19} />} label="Export" onClick={() => loadCode()} />
        </nav>
      </aside>

      <main className="min-w-0 flex-1">
        <header className="flex min-h-[92px] flex-wrap items-center justify-between gap-4 border-b-2 border-black px-5 py-5 md:px-10">
          <div>
            <div className="text-xs font-bold uppercase tracking-[0.2em]">Local data operations desk</div>
            <h1 className="mt-1 text-3xl font-bold uppercase tracking-[-0.06em] md:text-5xl">DATADOC</h1>
          </div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase">
            <span className="h-3 w-3 rounded-full border-2 border-black bg-green-400" />
            fitted-pipeline architecture
          </div>
        </header>

        {error && (
          <div className="mx-5 mt-5 flex items-start gap-3 border-2 border-black bg-red-200 p-4 text-sm font-bold md:mx-10">
            <X size={18} className="shrink-0" /> <span>{error}</span>
          </div>
        )}

        <div className="grid gap-6 p-5 md:p-10 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.6fr)]">
          <section className="min-w-0">
            <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b-2 border-black pb-5">
              <div>
                <p className="text-xs font-bold uppercase text-gray-600">Dataset workspace</p>
                <h2 className="mt-1 text-3xl font-bold uppercase tracking-[-0.04em]">{view === 'code' ? 'Exported pipeline' : view === 'pipeline' ? 'Plan and fit' : 'Profile and prepare'}</h2>
              </div>
              <button onClick={refreshProfile} className="flex items-center gap-2 border-2 border-black bg-white px-3 py-2 text-xs font-bold uppercase hover:bg-black hover:text-white"><RefreshCw size={15} /> Refresh</button>
            </div>

            {view !== 'code' && (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Metric label="Rows" value={profile?.rows?.toLocaleString()} tone="bg-bg-accent-yellow" />
                  <Metric label="Columns" value={profile?.columns} tone="bg-white" />
                  <Metric label="Findings" value={findings.length} tone="bg-bg-accent-pink" />
                  <Metric label="Pipeline" value={fitState ? 'FITTED' : 'DRAFT'} tone="bg-bg-accent-blue" />
                </div>

                <div className="mt-6 border-2 border-black bg-white p-5">
                  <div className="flex items-center gap-3 border-b-2 border-black pb-3"><ShieldCheck size={21} /><h3 className="font-bold uppercase">Preparation settings</h3></div>
                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    <label className="text-xs font-bold uppercase">Target column
                      <input value={config.target} onChange={(event) => setConfig({ ...config, target: event.target.value })} placeholder="Optional for cleaning; e.g. churn" className="mt-2 w-full border-2 border-black bg-bg-workspace p-3 font-mono text-sm outline-none focus:bg-yellow-50" />
                    </label>
                    <label className="text-xs font-bold uppercase">Task
                      <select value={config.task} onChange={(event) => setConfig({ ...config, task: event.target.value })} className="mt-2 w-full border-2 border-black bg-bg-workspace p-3 font-mono text-sm">
                        <option value="auto">Auto-detect</option><option value="classification">Classification</option><option value="regression">Regression</option>
                      </select>
                    </label>
                    <label className="text-xs font-bold uppercase">Scaling
                      <select value={config.scaling} onChange={(event) => setConfig({ ...config, scaling: event.target.value })} className="mt-2 w-full border-2 border-black bg-bg-workspace p-3 font-mono text-sm">
                        <option value="auto">Auto by estimator</option><option value="none">None</option><option value="standard">Standard</option><option value="robust">Robust</option>
                      </select>
                    </label>
                    <div className="flex flex-col justify-end gap-2 text-xs font-bold uppercase">
                      <label className="flex items-center gap-2"><input type="checkbox" checked={config.drop_identifiers} onChange={(event) => setConfig({ ...config, drop_identifiers: event.target.checked })} /> Drop suspected identifiers</label>
                      <label className="flex items-center gap-2"><input type="checkbox" checked={config.clip_outliers} onChange={(event) => setConfig({ ...config, clip_outliers: event.target.checked })} /> Enable IQR clipping</label>
                    </div>
                  </div>
                  <div className="mt-5 flex flex-wrap gap-3">
                    <button onClick={createPlan} disabled={busy} className="flex items-center gap-2 border-2 border-black bg-bg-accent-yellow px-4 py-3 text-xs font-bold uppercase hover:bg-black hover:text-white disabled:opacity-50"><ArrowRight size={16} /> {busy ? 'Working...' : 'Build plan'}</button>
                    <button onClick={fitPipeline} disabled={busy} className="flex items-center gap-2 border-2 border-black bg-black px-4 py-3 text-xs font-bold uppercase text-white hover:bg-gray-700 disabled:opacity-50"><Play size={16} /> Fit pipeline</button>
                  </div>
                  <p className="mt-4 text-xs text-gray-600">Fitting learns statistics from the loaded dataset. For model evaluation, fit on a training split and transform validation/test data with the saved artifact.</p>
                </div>
              </>
            )}

            {view === 'overview' && <Overview profile={profile} roles={roles} findings={findings} />}
            {view === 'pipeline' && <PipelineView plan={plan} fitState={fitState} preview={preview} outputColumns={outputColumns} />}
            {view === 'code' && <CodeView code={code} onLoad={loadCode} />}
          </section>

          <aside className="min-w-0">
            <ChatPanel history={chatHistory} input={chatInput} setInput={setChatInput} onSubmit={sendChatMessage} />
            <div className="mt-6 border-2 border-black bg-bg-accent-peach p-5">
              <div className="flex items-center gap-3 border-b-2 border-black pb-3"><FileJson size={19} /><h3 className="font-bold uppercase">Artifact actions</h3></div>
              <p className="mt-3 text-xs leading-5">Once fitted, export the transformed CSV or the reproducible Python wrapper generated from the fitted JSON state.</p>
              <div className="mt-4 grid gap-2">
                <button onClick={downloadCsv} className="flex items-center justify-between border-2 border-black bg-white px-3 py-3 text-left text-xs font-bold uppercase hover:bg-black hover:text-white">Download transformed CSV <Download size={15} /></button>
                <button onClick={loadCode} className="flex items-center justify-between border-2 border-black bg-white px-3 py-3 text-xs font-bold uppercase hover:bg-black hover:text-white">Load Python export <Code2 size={15} /></button>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function NavButton({ active, icon, label, onClick }) {
  return <button onClick={onClick} className={`flex w-full flex-col items-center gap-2 px-2 py-4 text-[10px] font-bold uppercase transition-colors ${active ? 'bg-black text-white' : 'hover:bg-black hover:text-white'}`}>{icon}<span>{label}</span></button>;
}

function Overview({ profile, roles, findings }) {
  return (
    <div className="mt-6 grid gap-6 lg:grid-cols-2">
      <div className="border-2 border-black bg-bg-accent-blue p-5">
        <h3 className="border-b-2 border-black pb-3 font-bold uppercase">Column roles</h3>
        <div className="mt-4 space-y-2">{roles.map((role) => <div key={role.name} className="flex items-center justify-between gap-3 border-b border-black/30 py-2 text-xs"><span className="truncate font-bold">{role.name}</span><Pill tone={role.role.includes('feature') ? 'bg-white' : 'bg-bg-accent-yellow'}>{role.role}</Pill></div>)}</div>
      </div>
      <div className="border-2 border-black bg-bg-accent-pink p-5">
        <h3 className="border-b-2 border-black pb-3 font-bold uppercase">Findings</h3>
        <div className="mt-4 space-y-3">{findings.length ? findings.map((finding, index) => <div key={`${finding.code}-${index}`} className="border-2 border-black bg-white p-3 text-xs"><div className="flex justify-between gap-2 font-bold uppercase"><span>{finding.code}</span><span>{finding.severity}</span></div><p className="mt-2 leading-5">{finding.message}</p></div>) : <p className="text-sm">No profile findings yet.</p>}</div>
      </div>
    </div>
  );
}

function PipelineView({ plan, fitState, preview, outputColumns }) {
  return (
    <div className="mt-6 space-y-6">
      <div className="border-2 border-black bg-bg-accent-yellow p-5"><h3 className="border-b-2 border-black pb-3 font-bold uppercase">Transformation plan</h3>{plan?.operations?.length ? <div className="mt-4 space-y-2">{plan.operations.map((operation, index) => <div key={`${operation.column}-${index}`} className="flex gap-3 border-b border-black/30 py-3 text-xs"><span className="font-bold">{String(index + 1).padStart(2, '0')}</span><div><strong>{operation.operation}</strong><div className="mt-1">{operation.column} — {operation.reason}</div></div></div>)}</div> : <p className="mt-4 text-sm">Build a plan to see proposed operations.</p>}</div>
      <div className="border-2 border-black bg-white p-5"><div className="flex items-center justify-between gap-3 border-b-2 border-black pb-3"><h3 className="font-bold uppercase">Fitted output schema</h3>{fitState && <Pill tone="bg-green-300"><Check size={12} className="mr-1" /> Fitted</Pill>}</div>{outputColumns.length ? <div className="mt-4 flex flex-wrap gap-2">{outputColumns.map((column) => <Pill key={column}>{column}</Pill>)}</div> : <p className="mt-4 text-sm">Fit the pipeline to generate the output schema.</p>}</div>
      {preview && <div className="overflow-hidden border-2 border-black bg-bg-workspace p-5"><h3 className="border-b-2 border-black pb-3 font-bold uppercase">Preview rows</h3><pre className="mt-4 overflow-auto text-xs leading-5">{JSON.stringify(preview.rows, null, 2)}</pre></div>}
    </div>
  );
}

function CodeView({ code, onLoad }) {
  return <div className="mt-6 border-2 border-black bg-black p-5 text-green-300"><div className="flex items-center justify-between gap-3 border-b border-green-300/50 pb-3"><h3 className="font-bold uppercase">Reproducible Python export</h3>{!code && <button onClick={onLoad} className="border-2 border-green-300 px-3 py-2 text-xs font-bold uppercase hover:bg-green-300 hover:text-black">Load export</button>}</div><pre className="mt-5 max-h-[62vh] overflow-auto whitespace-pre-wrap text-xs leading-5">{code || 'Fit a pipeline, then load the export.'}</pre></div>;
}

function ChatPanel({ history, input, setInput, onSubmit }) {
  return <div className="border-2 border-black bg-white p-5"><div className="flex items-center gap-3 border-b-2 border-black pb-3"><MessageSquare size={19} /><h3 className="font-bold uppercase">Optional AI assistant</h3></div><p className="mt-3 text-xs leading-5 text-gray-600">AI can explain metadata and suggest registered transformations. It never executes generated code.</p><div className="mt-4 max-h-56 space-y-3 overflow-y-auto">{history.map((message, index) => <div key={index} className={`border-2 border-black p-3 text-xs leading-5 ${message.role === 'user' ? 'bg-bg-accent-peach' : 'bg-bg-workspace'}`}><span className="font-bold uppercase">{message.role}</span><p className="mt-1 whitespace-pre-wrap">{message.content}</p></div>)}</div><form onSubmit={onSubmit} className="mt-4 flex gap-2"><input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about the dataset..." className="min-w-0 flex-1 border-2 border-black bg-bg-workspace px-3 py-2 text-xs outline-none focus:bg-yellow-50" /><button className="border-2 border-black bg-black px-3 py-2 text-white hover:bg-gray-700"><Terminal size={15} /></button></form></div>;
}

export default App;
