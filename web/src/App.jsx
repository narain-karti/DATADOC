import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Database, Zap, Code, LogOut, Terminal, Activity, FileJson, Download, Wand2 } from 'lucide-react';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const API_BASE = "http://localhost:8000/api";

function App() {
  const [activeTab, setActiveTab] = useState('workspace'); // 'workspace', 'code'
  const [metadata, setMetadata] = useState(null);
  const [plugins, setPlugins] = useState([
    { id: 'MissingValuePlugin', name: 'Missing Value Imputer', color: 'bg-accent-yellow' },
    { id: 'OutlierPlugin', name: 'Outlier Clipper', color: 'bg-accent-pink' },
    { id: 'DatetimePlugin', name: 'Datetime Extractor', color: 'bg-accent-blue' },
    { id: 'CategoricalEncoderPlugin', name: 'Categorical Encoder', color: 'bg-accent-peach' },
    { id: 'ScalingPlugin', name: 'Standard Scaler', color: 'bg-accent-orange' },
  ]);
  const [pipelineResults, setPipelineResults] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [codeContent, setCodeContent] = useState("");

  useEffect(() => {
    fetchMetadata();
  }, []);

  const fetchMetadata = async () => {
    try {
      const res = await axios.get(`${API_BASE}/dataset/metadata`);
      setMetadata(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchCode = async () => {
    try {
      const res = await axios.get(`${API_BASE}/dataset/export/code`);
      setCodeContent(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const handleDragEnd = (result) => {
    if (!result.destination) return;
    const items = Array.from(plugins);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    setPlugins(items);
  };

  const runPipeline = async () => {
    setLoading(true);
    try {
      const pNames = plugins.map(p => p.id);
      const res = await axios.post(`${API_BASE}/dataset/plugins`, { plugins: pNames });
      setPipelineResults(res.data);
      if (activeTab === 'code') fetchCode();
    } catch (e) {
      console.error(e);
      alert("Error running pipeline");
    } finally {
      setLoading(false);
    }
  };

  const autoRecommend = async () => {
    try {
      const res = await axios.get(`${API_BASE}/dataset/recommend`);
      const recommendedPlugins = res.data.plugins;
      // Filter out plugins that don't need to trigger, and reorder based on priority
      const activeP = recommendedPlugins.filter(p => p.will_trigger).sort((a,b) => a.priority - b.priority);
      
      const newPluginList = activeP.map(p => {
        const existing = plugins.find(ep => ep.id === p.name);
        return existing || { id: p.name, name: p.name, color: 'bg-accent-yellow' };
      });
      if (newPluginList.length > 0) {
        setPlugins(newPluginList);
      } else {
        alert("No transformations recommended based on dataset health!");
      }
    } catch (e) {
      console.error(e);
    }
  };

  const downloadCSV = () => {
    window.location.href = `${API_BASE}/dataset/export/csv`;
  };

  const downloadCode = () => {
    window.location.href = `${API_BASE}/dataset/export/code`;
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput;
    setChatInput("");
    setChatHistory([...chatHistory, { role: 'user', content: msg }]);
    
    try {
      const res = await axios.post(`${API_BASE}/agent/chat`, { message: msg });
      setChatHistory(prev => [...prev, { role: 'assistant', content: res.data.response }]);
    } catch (e) {
      console.error(e);
      setChatHistory(prev => [...prev, { role: 'assistant', content: "Error connecting to AI Agent." }]);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden font-mono bg-bg-workspace text-black selection:bg-black selection:text-white">
      {/* Sidebar - Mint Green */}
      <div className="w-[100px] bg-bg-sidebar border-r-2 border-black flex flex-col items-center py-6 justify-between shrink-0 z-10">
        <div className="flex flex-col gap-6 w-full items-center">
          <div className="w-12 h-12 bg-black text-white flex items-center justify-center font-bold text-xl mb-4">
            DD
          </div>
          
          <button 
            onClick={() => setActiveTab('workspace')}
            className={`w-full py-4 flex flex-col items-center gap-1 transition-colors ${activeTab === 'workspace' ? 'bg-black text-white' : 'hover:bg-black hover:text-white'}`}
          >
            <Database size={24} />
            <span className="text-xs uppercase font-bold mt-1">Data</span>
          </button>
          
          <button 
            onClick={() => { setActiveTab('code'); fetchCode(); }}
            className={`w-full py-4 flex flex-col items-center gap-1 transition-colors ${activeTab === 'code' ? 'bg-black text-white' : 'hover:bg-black hover:text-white'}`}
          >
            <Code size={24} />
            <span className="text-xs uppercase font-bold mt-1">Code</span>
          </button>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <header className="h-[80px] border-b-2 border-black flex items-center justify-between px-8 bg-bg-workspace shrink-0 z-10">
          <div>
            <h1 className="text-3xl font-bold uppercase tracking-tighter">DATADOC Dashboard</h1>
            <p className="text-sm text-red-600 font-bold bg-yellow-200 px-2 inline-block border-2 border-black mt-1">⚠️ UI DEPRECATED: PENDING REWRITE FOR LEAKAGE-SAFE PIPELINES ⚠️</p>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={downloadCSV} className="flex items-center gap-2 border-2 border-black px-4 py-2 bg-bg-accent-yellow hover:bg-yellow-300 font-bold uppercase text-sm">
              <Download size={18} /> Export CSV
            </button>
            <button onClick={downloadCode} className="flex items-center gap-2 border-2 border-black px-4 py-2 bg-white hover:bg-gray-100 font-bold uppercase text-sm">
              <Download size={18} /> Export Code
            </button>
            <div className="flex items-center gap-2 border-2 border-black px-4 py-2 bg-white ml-4">
              <Activity size={18} />
              <span className="font-bold text-sm">Status: Online</span>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden">
          
          {activeTab === 'workspace' ? (
            <>
              {/* Left Column: Data & Plugins */}
              <div className="w-1/2 border-r-2 border-black overflow-y-auto p-8 flex flex-col gap-8 custom-scrollbar relative">
                
                {/* Metadata Card - Golden Yellow */}
                <div className="bg-bg-accent-yellow border-2 border-black p-6 relative">
                  <h2 className="text-2xl font-bold uppercase mb-4 border-b-2 border-black pb-2 inline-block">Dataset Info</h2>
                  {metadata ? (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white border-2 border-black p-4 text-center">
                        <p className="text-xs uppercase font-bold mb-1">Rows</p>
                        <p className="text-3xl font-bold">{metadata.rows.toLocaleString()}</p>
                      </div>
                      <div className="bg-white border-2 border-black p-4 text-center">
                        <p className="text-xs uppercase font-bold mb-1">Columns</p>
                        <p className="text-3xl font-bold">{metadata.columns}</p>
                      </div>
                      <div className="col-span-2 bg-white border-2 border-black p-4">
                        <p className="text-xs uppercase font-bold mb-1 flex items-center gap-2"><FileJson size={16}/> File Path</p>
                        <p className="text-sm truncate">{metadata.file}</p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm">Loading metadata...</p>
                  )}
                </div>

                {/* Plugin Orchestrator - Lavender Pink */}
                <div className="bg-bg-accent-pink border-2 border-black p-6">
                  <div className="flex justify-between items-end mb-6 border-b-2 border-black pb-2">
                    <h2 className="text-2xl font-bold uppercase">Pipeline</h2>
                    <div className="flex gap-2">
                      <button 
                        onClick={autoRecommend}
                        className="bg-white text-black border-2 border-black px-4 py-2 uppercase font-bold text-sm hover:bg-gray-100 flex items-center gap-2"
                      >
                        <Wand2 size={16} /> Auto-Recommend
                      </button>
                      <button 
                        onClick={runPipeline}
                        disabled={loading}
                        className="bg-black text-white px-6 py-2 uppercase font-bold text-sm hover:bg-gray-800 transition-colors disabled:opacity-50"
                      >
                        {loading ? 'Running...' : 'Execute'}
                      </button>
                    </div>
                  </div>

                  <DragDropContext onDragEnd={handleDragEnd}>
                    <Droppable droppableId="plugins">
                      {(provided) => (
                        <div {...provided.droppableProps} ref={provided.innerRef} className="flex flex-col gap-3">
                          {plugins.map((plugin, index) => (
                            <Draggable key={plugin.id} draggableId={plugin.id} index={index}>
                              {(provided) => (
                                <div
                                  ref={provided.innerRef}
                                  {...provided.draggableProps}
                                  {...provided.dragHandleProps}
                                  className={`border-2 border-black p-4 ${plugin.color} font-bold text-lg flex items-center justify-between group hover:-translate-y-1 transition-transform`}
                                >
                                  <span>{index + 1}. {plugin.name}</span>
                                  <div className="w-6 h-6 border-2 border-black flex items-center justify-center bg-white">
                                    <span className="text-xs">≡</span>
                                  </div>
                                </div>
                              )}
                            </Draggable>
                          ))}
                          {provided.placeholder}
                        </div>
                      )}
                    </Droppable>
                  </DragDropContext>
                </div>
                
                {/* Pipeline Results */}
                {pipelineResults && (
                  <div className="bg-bg-accent-blue border-2 border-black p-6">
                    <h2 className="text-xl font-bold uppercase mb-4 border-b-2 border-black pb-2">Results & Diffs</h2>
                    
                    {pipelineResults.diff && (
                      <div className="mb-6 bg-white border-2 border-black p-4">
                        <h3 className="font-bold uppercase text-sm mb-4">Missing Values Comparison</h3>
                        <div className="h-48">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={[
                              { name: 'Original', missing: pipelineResults.diff.original_missing },
                              { name: 'Cleaned', missing: pipelineResults.diff.clean_missing }
                            ]}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#000" />
                              <XAxis dataKey="name" stroke="#000" tick={{fontFamily: 'Space Mono', fontWeight: 'bold'}} />
                              <YAxis stroke="#000" tick={{fontFamily: 'Space Mono'}} />
                              <Tooltip cursor={{fill: 'rgba(0,0,0,0.1)'}} contentStyle={{border: '2px solid #000', borderRadius: '0', fontFamily: 'Space Mono', fontWeight: 'bold', backgroundColor: '#F3EAD5'}} />
                              <Bar dataKey="missing" fill="#F4C542" stroke="#000" strokeWidth={2} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}
                    
                    <h3 className="font-bold uppercase text-sm mb-2">Execution Logs</h3>
                    <div className="flex flex-col gap-2">
                      {pipelineResults.results.map((r, i) => (
                        <div key={i} className="bg-white border-2 border-black p-3 text-sm">
                          <span className="font-bold">{r.plugin}:</span> {r.status}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: AI Chat */}
              <div className="w-1/2 bg-white flex flex-col custom-scrollbar relative">
                <div className="p-6 border-b-2 border-black bg-bg-workspace flex items-center gap-3 shrink-0">
                  <Terminal size={28} />
                  <h2 className="text-2xl font-bold uppercase">Agentic Engineer</h2>
                </div>
                
                <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-6 bg-white">
                  {chatHistory.length === 0 && (
                    <div className="text-center text-gray-500 mt-10">
                      <p className="mb-2">System initialized.</p>
                      <p>Ask the AI to analyze data or write a custom pandas script.</p>
                    </div>
                  )}
                  {chatHistory.map((msg, i) => (
                    <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                      <div className="text-xs font-bold uppercase mb-1">{msg.role}</div>
                      <div className={`p-4 border-2 border-black max-w-[85%] whitespace-pre-wrap text-sm ${msg.role === 'user' ? 'bg-bg-accent-peach' : 'bg-bg-workspace'}`}>
                        {msg.content}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="p-6 border-t-2 border-black bg-bg-workspace shrink-0">
                  <div className="flex gap-4">
                    <input 
                      type="text" 
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && sendChatMessage()}
                      placeholder="Tell the AI what to do..."
                      className="flex-1 border-2 border-black p-3 bg-white outline-none focus:bg-yellow-50 font-mono"
                    />
                    <button 
                      onClick={sendChatMessage}
                      className="bg-black text-white px-6 font-bold uppercase hover:bg-gray-800 transition-colors"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* Code Viewer Tab */
            <div className="flex-1 flex flex-col p-8 overflow-y-auto bg-white">
              <div className="bg-bg-accent-peach border-2 border-black p-6 mb-6">
                <h2 className="text-2xl font-bold uppercase">Generated Python Pipeline</h2>
                <p className="text-sm mt-2">This is the underlying Polars script that represents your configured pipeline.</p>
              </div>
              <div className="flex-1 border-2 border-black bg-bg-workspace p-6 overflow-auto">
                <pre className="font-mono text-sm whitespace-pre-wrap">{codeContent || "Execute the pipeline first to generate code!"}</pre>
              </div>
            </div>
          )}
          
        </div>
      </div>
    </div>
  );
}

export default App;
