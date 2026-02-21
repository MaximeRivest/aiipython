"""Standalone HTTP traffic inspector server.

Runs a background HTTP server that serves a live HTML viewer
(with SSE push) showing all LM API calls captured by :mod:`wire`.

The server is fully self-contained — no Pi extension needed.
"""

from __future__ import annotations

import json
import queue
import socket
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# ── state ───────────────────────────────────────────────────────────

_entries: list[dict[str, Any]] = []
_next_id = 0
_id_lock = threading.Lock()
_sse_queues: list[queue.Queue[str]] = []
_sse_lock = threading.Lock()
_server: HTTPServer | None = None
_server_port: int = 0
_server_thread: threading.Thread | None = None


def _next_entry_id() -> int:
    global _next_id
    with _id_lock:
        _next_id += 1
        return _next_id


def _send_sse(event_type: str, data: Any) -> None:
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead: list[queue.Queue] = []
        for q in _sse_queues:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


# ── external entry map (wire_bridge pushes here) ────────────────────

_ext_map: dict[int, dict[str, Any]] = {}  # wire entry_id → our entry


def ingest_request(
    entry_id: int,
    model: str,
    messages: list[dict],
    kwargs: dict | None = None,
) -> None:
    eid = _next_entry_id()
    # Build a fake request body so the conversation tab works
    try:
        req_body = json.dumps({"model": model, "messages": messages, **(kwargs or {})})
    except Exception:
        req_body = None

    entry: dict[str, Any] = {
        "id": eid,
        "timestamp": int(time.time() * 1000),
        "method": "POST",
        "url": model,
        "requestHeaders": {},
        "requestBody": req_body,
        "responseStatus": None,
        "responseStatusText": None,
        "responseHeaders": {},
        "responseBody": "",
        "duration": None,
        "error": None,
        "isStreaming": True,
        "complete": False,
        "source": "python",
    }
    _ext_map[entry_id] = entry
    _entries.append(entry)
    if len(_entries) > 500:
        _entries.pop(0)
    _send_sse("entry", entry)


def ingest_chunk(entry_id: int, chunk: str) -> None:
    entry = _ext_map.get(entry_id)
    if not entry:
        return
    entry["responseBody"] += chunk
    _send_sse("entry", entry)


def ingest_done(
    entry_id: int,
    full_response: str = "",
    duration_ms: float = 0,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    entry = _ext_map.pop(entry_id, None)
    if not entry:
        return
    if full_response and not entry["responseBody"]:
        entry["responseBody"] = full_response
    entry["duration"] = duration_ms or None
    entry["responseStatus"] = 500 if error else 200
    entry["responseStatusText"] = "Error" if error else "OK"
    entry["error"] = error
    entry["complete"] = True
    if usage:
        entry["responseHeaders"]["x-usage"] = json.dumps(usage)
    _send_sse("entry", entry)


def ingest_error(entry_id: int, error: str, duration_ms: float = 0) -> None:
    ingest_done(entry_id, error=error, duration_ms=duration_ms)


# ── HTTP handler ────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a: Any) -> None:
        pass  # silence request logging

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            body = _get_viewer_html().encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/traffic":
            body = json.dumps(_entries).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/events":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b"event: connected\ndata: {}\n\n")
            self.wfile.flush()

            q: queue.Queue[str] = queue.Queue(maxsize=1000)
            with _sse_lock:
                _sse_queues.append(q)
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # keepalive
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _sse_lock:
                    if q in _sse_queues:
                        _sse_queues.remove(q)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/clear":
            _entries.clear()
            _ext_map.clear()
            _send_sse("clear", {})
            body = b'{"ok":true}'
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)


# ── server lifecycle ────────────────────────────────────────────────

def get_port() -> int:
    """Return the inspector port, or 0 if not running."""
    return _server_port


def is_running() -> bool:
    return _server is not None


def start(*, open_browser: bool = False) -> int:
    """Start the traffic inspector server. Returns the port."""
    global _server, _server_port, _server_thread

    if _server is not None:
        if open_browser:
            _open_browser(f"http://127.0.0.1:{_server_port}")
        return _server_port

    # Pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = HTTPServer(("127.0.0.1", port), _Handler)
    srv.timeout = 0.5

    def run() -> None:
        while _server is srv:
            srv.handle_request()

    _server = srv
    _server_port = port
    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()

    if open_browser:
        _open_browser(f"http://127.0.0.1:{port}")

    return port


def stop() -> None:
    global _server, _server_port, _server_thread
    srv = _server
    _server = None
    _server_port = 0
    _server_thread = None
    if srv:
        srv.shutdown()


def _open_browser(url: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            subprocess.Popen(["start", url], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── HTML viewer (self-contained) ────────────────────────────────────

def _get_viewer_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>aiipython Traffic Inspector</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #1c2129; --bgh: #1f2937;
    --border: #30363d; --borderl: #3d444d;
    --text: #e6edf3; --text2: #8b949e; --textm: #6e7681;
    --accent: #58a6ff; --success: #3fb950; --warn: #d29922; --err: #f85149;
    --jk: #79c0ff; --js: #a5d6ff; --jn: #79c0ff; --jb: #ff7b72; --jnull: #8b949e;
    --mono: 'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',Consolas,monospace;
    --sans: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
    --r: 6px;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:var(--sans);background:var(--bg);color:var(--text);font-size:13px;line-height:1.5;overflow:hidden;height:100vh}
  ::-webkit-scrollbar{width:8px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
  .app{display:flex;flex-direction:column;height:100vh}
  .hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;gap:16px;flex-shrink:0}
  .hdr-logo{font-size:18px;font-weight:700;white-space:nowrap}.hdr-logo span{color:var(--accent)}
  .hdr-stats{color:var(--text2);font-size:12px;display:flex;gap:12px}
  .badge{background:var(--bg3);padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid var(--border)}
  .hdr-acts{margin-left:auto;display:flex;gap:8px;align-items:center}
  .btn{background:var(--bg3);border:1px solid var(--border);color:var(--text2);padding:4px 12px;border-radius:var(--r);cursor:pointer;font-size:12px;transition:all .15s;display:flex;align-items:center;gap:4px}
  .btn:hover{background:var(--bgh);color:var(--text)}.btn-d:hover{background:#3d1214;color:var(--err)}.btn-a{background:var(--accent);color:#000;border-color:var(--accent)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--success);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .fbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:8px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex-shrink:0}
  .mf,.sf{display:flex;gap:4px}
  .mbtn,.sbtn{background:var(--bg3);border:1px solid var(--border);color:var(--textm);padding:2px 8px;border-radius:var(--r);cursor:pointer;font-size:11px;font-weight:600;text-transform:uppercase;transition:all .15s}
  .mbtn:hover,.sbtn:hover{border-color:var(--borderl);color:var(--text2)}
  .mbtn.a{background:var(--accent);border-color:var(--accent);color:#000}
  .sbtn.a{background:var(--accent);border-color:var(--accent);color:#000}
  .sbox{flex:1;min-width:200px;position:relative}
  .sbox input{width:100%;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:4px 10px 4px 28px;border-radius:var(--r);font-size:12px;outline:none}
  .sbox input:focus{border-color:var(--accent)}.sbox input::placeholder{color:var(--textm)}
  .sbox::before{content:'🔍';position:absolute;left:8px;top:50%;transform:translateY(-50%);font-size:11px;pointer-events:none}
  .tlist{flex:1;overflow-y:auto}
  .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--textm);gap:12px}
  .empty-icon{font-size:48px;opacity:.3}
  .ent{border-bottom:1px solid var(--border)}.ent:hover>.ehdr{background:var(--bgh)}.ent.exp>.ehdr{background:var(--bg3)}
  .ehdr{display:grid;grid-template-columns:32px 56px 1fr 70px 80px 60px 120px;align-items:center;padding:6px 16px;cursor:pointer;user-select:none;gap:8px;min-height:36px}
  .eexp{color:var(--textm);font-size:10px;transition:transform .15s;text-align:center}.ent.exp .eexp{transform:rotate(90deg)}
  .emeth{font-size:11px;font-weight:700;font-family:var(--mono);padding:1px 6px;border-radius:3px;text-align:center}
  .emeth.POST{background:rgba(88,166,255,.15);color:var(--accent)}
  .emeth.GET{background:rgba(63,185,80,.15);color:var(--success)}
  .eurl{font-family:var(--mono);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .esrc{font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;background:rgba(63,185,80,.15);color:var(--success)}
  .estat{font-family:var(--mono);font-size:12px;font-weight:600;text-align:center}
  .estat.s2{color:var(--success)}.estat.pend{color:var(--textm)}.estat.err{color:var(--err)}
  .edur{font-family:var(--mono);font-size:12px;text-align:right}
  .edur.fast{color:var(--success)}.edur.med{color:var(--warn)}.edur.slow{color:var(--err)}
  .esz{font-family:var(--mono);font-size:11px;color:var(--textm);text-align:right}
  .etm{font-size:11px;color:var(--textm);text-align:right;font-family:var(--mono)}
  .estr{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);margin-left:4px;animation:pulse 1s infinite}
  .edet{display:none;background:var(--bg2);border-top:1px solid var(--border)}.ent.exp .edet{display:block}
  .dtabs{display:flex;border-bottom:1px solid var(--border);background:var(--bg3);padding:0 16px}
  .dtab{padding:8px 16px;font-size:12px;color:var(--text2);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
  .dtab:hover{color:var(--text)}.dtab.a{color:var(--accent);border-bottom-color:var(--accent)}
  .dpan{display:none;padding:12px 16px;max-height:600px;overflow-y:auto}.dpan.a{display:block}
  .hsec{margin-bottom:16px}.htit{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text2);margin-bottom:6px}
  .htbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
  .htbl td{padding:3px 8px;border-bottom:1px solid var(--border);vertical-align:top}
  .htbl td:first-child{color:var(--jk);white-space:nowrap;width:220px;font-weight:600}
  .htbl td:last-child{word-break:break-all}
  .cblk{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:12px;font-family:var(--mono);font-size:12px;line-height:1.6;overflow-x:auto;white-space:pre-wrap;word-break:break-word;max-height:500px;overflow-y:auto;position:relative}
  .cblk .cpb{position:sticky;float:right;top:0;right:0;background:var(--bg3);border:1px solid var(--border);color:var(--text2);padding:2px 8px;border-radius:var(--r);cursor:pointer;font-size:11px;z-index:1}
  .cblk .cpb:hover{color:var(--text);background:var(--bgh)}
  .jk{color:var(--jk)}.jstr{color:var(--js)}.jnum{color:var(--jn)}.jbool{color:var(--jb)}.jnul{color:var(--jnull)}
  .conv{display:flex;flex-direction:column;gap:12px}
  .cmsg{padding:10px 14px;border-radius:var(--r);max-width:95%;font-size:13px;line-height:1.6}
  .cmsg.sys{background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.2);align-self:stretch;max-width:100%}
  .cmsg.usr{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.2);align-self:flex-end}
  .cmsg.ast{background:rgba(188,140,255,.08);border:1px solid rgba(188,140,255,.2);align-self:flex-start}
  .cmsg.tl{background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.2);font-family:var(--mono);font-size:12px;align-self:stretch;max-width:100%}
  .crole{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
  .cmsg.sys .crole{color:var(--accent)}.cmsg.usr .crole{color:var(--success)}.cmsg.ast .crole{color:#bc8cff}.cmsg.tl .crole{color:var(--warn)}
  .ccont{white-space:pre-wrap;word-break:break-word}
  .ccoll{cursor:pointer;user-select:none}
  .ccbody{display:none;margin-top:6px}.ccbody.open{display:block}
  .ctog{font-size:11px;color:var(--textm);margin-left:8px}
  .nobody{color:var(--textm);padding:20px;text-align:center;font-style:italic}
  .ftr{background:var(--bg2);border-top:1px solid var(--border);padding:6px 16px;display:flex;justify-content:space-between;font-size:11px;color:var(--textm);flex-shrink:0}
  .ftr kbd{background:var(--bg3);border:1px solid var(--border);padding:0 4px;border-radius:3px;font-family:var(--mono);font-size:10px}
  @media(max-width:900px){.ehdr{grid-template-columns:24px 48px 1fr 56px 64px}.esz,.etm{display:none}}
</style>
</head>
<body>
<div class="app">
  <div class="hdr">
    <div class="hdr-logo"><span>🐍</span> aiipython Traffic Inspector</div>
    <div class="hdr-stats">
      <span class="badge" id="st-tot">0 requests</span>
      <span class="badge" id="st-str" style="display:none">0 streaming</span>
      <span class="badge" id="st-err" style="display:none">0 errors</span>
    </div>
    <div class="hdr-acts">
      <div class="dot" id="dot"></div>
      <button class="btn btn-a" id="btn-as">↓ Auto-scroll</button>
      <button class="btn btn-d" id="btn-clr">✕ Clear</button>
    </div>
  </div>
  <div class="fbar">
    <div class="mf">
      <button class="mbtn a" data-m="ALL">All</button>
      <button class="mbtn" data-m="POST">POST</button>
      <button class="mbtn" data-m="GET">GET</button>
    </div>
    <div class="sbox"><input id="search" placeholder="Filter by model, body content..."></div>
    <div class="sf">
      <button class="sbtn a" data-s="ALL">All</button>
      <button class="sbtn" data-s="2xx">2xx</button>
      <button class="sbtn" data-s="err">Err</button>
    </div>
  </div>
  <div class="tlist" id="tlist">
    <div class="empty" id="empty">
      <div class="empty-icon">🐍</div>
      <div>No LM traffic captured yet</div>
      <div style="font-size:12px">Traffic appears here as aiipython makes API calls</div>
    </div>
  </div>
  <div class="ftr">
    <span><kbd>↑</kbd><kbd>↓</kbd> Navigate <kbd>Enter</kbd> Expand <kbd>C</kbd> Conversation <kbd>/</kbd> Search</span>
    <span id="fst">Connecting…</span>
  </div>
</div>
<script>
let A=[],F=[],EX=new Set(),SI=-1,AS=true,MF='ALL',SF='ALL',SQ='',DT={};
function E(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function fB(b){return b<1024?b+' B':b<1048576?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(1)+' MB'}
function fD(m){return m==null?'—':m<1000?Math.round(m)+'ms':m<60000?(m/1000).toFixed(1)+'s':(m/60000).toFixed(1)+'m'}
function dC(m){return m==null?'':m<1000?'fast':m<5000?'med':'slow'}
function sC(s){return s==null?'pend':s>=200&&s<300?'s2':s>=400?'err':''}
function fT(t){let d=new Date(t);return d.toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'})+'.'+String(d.getMilliseconds()).padStart(3,'0')}
function hJ(s){try{return sH(JSON.stringify(JSON.parse(s),null,2))}catch{return E(s)}}
function sH(j){j=E(j);return j.replace(/("(\\\\u[a-fA-F0-9]{4}|\\\\[^u]|[^\\\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,function(m){let c='jnum';if(/^"/.test(m))c=/:$/.test(m)?'jk':'jstr';else if(/true|false/.test(m))c='jbool';else if(/null/.test(m))c='jnul';return'<span class="'+c+'">'+m+'</span>'})}
function xConv(e){
  let ms=[];if(!e.requestBody)return ms;
  try{let b=JSON.parse(e.requestBody);
    if(b.model)ms.push({r:'info',c:'Model: '+b.model});
    if(b.system){let t=typeof b.system==='string'?b.system:JSON.stringify(b.system);ms.push({r:'system',c:t,cl:t.length>300})}
    if(b.messages)for(let m of b.messages){let role=m.role||'?',txt='';
      if(role==='system'){let t=typeof m.content==='string'?m.content:JSON.stringify(m.content);ms.push({r:'system',c:t,cl:t.length>300});continue}
      if(typeof m.content==='string')txt=m.content;
      else if(Array.isArray(m.content)){let ps=[];for(let p of m.content){if(typeof p==='string')ps.push(p);else if(p.type==='text')ps.push(p.text||'');else if(p.type==='image_url'||p.type==='image')ps.push('[image]');else ps.push('['+(p.type||'?')+']')}txt=ps.join('\n')}
      else txt=String(m.content||'');
      if(role==='tool'||role==='toolResult')ms.push({r:'tool',c:txt||'[tool]',cl:txt.length>200});
      else ms.push({r:role,c:txt||'[empty]',cl:txt.length>500})}
  }catch{}
  if(e.responseBody){ms.push({r:'assistant',c:e.responseBody,cl:e.responseBody.length>500})}
  return ms;
}
function aFilt(){F=A.filter(e=>{if(MF!=='ALL'&&e.method!==MF)return false;if(SF!=='ALL'){if(SF==='err'){if(!e.error&&e.responseStatus!==null)return false}else{if(e.responseStatus==null||String(e.responseStatus)[0]!=='2')return false}}if(SQ){let q=SQ.toLowerCase(),h=[e.url,e.method,e.requestBody||'',e.responseBody||'',e.error||''].join(' ').toLowerCase();if(!h.includes(q))return false}return true})}
function render(){
  let c=document.getElementById('tlist'),em=document.getElementById('empty');
  if(!F.length){c.querySelectorAll('.ent').forEach(e=>e.remove());if(em)em.style.display='';uSt();return}
  if(em)em.style.display='none';
  let ex={};c.querySelectorAll('.ent').forEach(e=>{ex[e.dataset.id]=e});
  let tgt=new Set(F.map(e=>String(e.id)));
  for(let[id,el]of Object.entries(ex))if(!tgt.has(id))el.remove();
  let pr=null;
  for(let e of F){let id=String(e.id),el=ex[id];
    if(el)updEl(el,e);else{el=mkEl(e);if(pr&&pr.nextSibling)c.insertBefore(el,pr.nextSibling);else if(!pr){let f=c.querySelector('.ent');if(f)c.insertBefore(el,f);else c.appendChild(el)}else c.appendChild(el)}
    pr=el}
  uSt();if(AS)c.scrollTop=c.scrollHeight}
function mkEl(e){
  let el=document.createElement('div');el.className='ent'+(EX.has(e.id)?' exp':'');el.dataset.id=String(e.id);
  let sc=e.error?'err':sC(e.responseStatus),dc=dC(e.duration),sz=(e.requestBody?.length||0)+e.responseBody.length;
  let st=!e.complete&&e.isStreaming?'<span class="estr"></span>':'';
  el.innerHTML='<div class="ehdr"><span class="eexp">▶</span><span class="emeth '+e.method+'">'+e.method+'</span><span class="eurl" title="'+E(e.url)+'">'+E(e.url)+'<span class="esrc">🐍</span></span><span class="estat '+sc+'">'+(e.error?'ERR':e.responseStatus||'...')+st+'</span><span class="edur '+dc+'">'+fD(e.duration)+'</span><span class="esz">'+fB(sz)+'</span><span class="etm">'+fT(e.timestamp)+'</span></div><div class="edet" id="det-'+e.id+'"></div>';
  el.querySelector('.ehdr').addEventListener('click',()=>tog(e.id));
  if(EX.has(e.id))rDet(el.querySelector('.edet'),e);return el}
function updEl(el,e){
  let sc=e.error?'err':sC(e.responseStatus),dc=dC(e.duration),sz=(e.requestBody?.length||0)+e.responseBody.length;
  let st=!e.complete&&e.isStreaming?'<span class="estr"></span>':'';
  let s=el.querySelector('.estat');if(s){s.className='estat '+sc;s.innerHTML=(e.error?'ERR':e.responseStatus||'...')+st}
  let d=el.querySelector('.edur');if(d){d.className='edur '+dc;d.textContent=fD(e.duration)}
  let z=el.querySelector('.esz');if(z)z.textContent=fB(sz);
  if(EX.has(e.id)){let dt=el.querySelector('.edet');if(dt)rDet(dt,e)}}
function tog(id){if(EX.has(id))EX.delete(id);else EX.add(id);let el=document.querySelector('.ent[data-id="'+id+'"]');if(!el)return;el.classList.toggle('exp');if(EX.has(id)){let e=A.find(x=>x.id===id);if(e)rDet(el.querySelector('.edet'),e)}}
function rDet(d,e){
  let at=DT[e.id]||'conv',isLLM=e.requestBody&&(e.requestBody.includes('"messages"')||e.requestBody.includes('"model"'));
  let tb='<div class="dtabs">';
  if(isLLM)tb+='<div class="dtab'+(at==='conv'?' a':'')+'" data-t="conv">💬 Conversation</div>';
  tb+='<div class="dtab'+(at==='req'?' a':'')+'" data-t="req">Request Body</div><div class="dtab'+(at==='res'?' a':'')+'" data-t="res">Response</div><div class="dtab'+(at==='hdr'?' a':'')+'" data-t="hdr">Info</div></div>';
  let cv='';
  if(isLLM){cv='<div class="dpan'+(at==='conv'?' a':'')+'" data-p="conv">';let ms=xConv(e);
    if(ms.length){cv+='<div class="conv">';for(let i=0;i<ms.length;i++){let m=ms[i],rc=m.r==='info'?'sys':m.r==='system'?'sys':m.r==='user'?'usr':m.r==='assistant'?'ast':m.r==='tool'?'tl':'sys';
      let rl=m.r==='info'?'📋 Info':m.r.charAt(0).toUpperCase()+m.r.slice(1),ct=E(m.c);
      cv+='<div class="cmsg '+rc+'"><div class="crole">'+rl;
      if(m.cl){cv+='<span class="ctog" data-i="'+i+'">[expand]</span></div><div class="ccont ccoll" data-i="'+i+'">'+ct.slice(0,200)+'…</div><div class="ccbody" data-b="'+i+'"><div class="ccont">'+ct+'</div></div>'}
      else{cv+='</div><div class="ccont">'+ct+'</div>'}
      cv+='</div>'}cv+='</div>'}else cv+='<div class="nobody">No conversation data</div>';cv+='</div>'}
  let rq='<div class="dpan'+(at==='req'?' a':'')+'" data-p="req">';
  rq+=e.requestBody?'<div class="cblk"><button class="cpb" onclick="cpTx(this)">Copy</button>'+hJ(e.requestBody)+'</div>':'<div class="nobody">No request body</div>';rq+='</div>';
  let rs='<div class="dpan'+(at==='res'?' a':'')+'" data-p="res">';
  if(e.responseBody){rs+='<div class="cblk"><button class="cpb" onclick="cpTx(this)">Copy</button>'+E(e.responseBody)+'</div>';if(!e.complete)rs+='<div style="color:var(--accent);padding:8px;font-size:12px">⏳ Streaming…</div>'}
  else rs+=e.complete?'<div class="nobody">No response</div>':'<div class="nobody">⏳ Waiting…</div>';rs+='</div>';
  let hd='<div class="dpan'+(at==='hdr'?' a':'')+'" data-p="hdr"><div class="hsec"><div class="htit">General</div><table class="htbl"><tbody>';
  hd+='<tr><td>Model</td><td>'+E(e.url)+'</td></tr><tr><td>Method</td><td>'+e.method+'</td></tr><tr><td>Status</td><td>'+(e.responseStatus||'Pending')+'</td></tr>';
  if(e.duration)hd+='<tr><td>Duration</td><td>'+fD(e.duration)+'</td></tr>';
  if(e.error)hd+='<tr><td>Error</td><td style="color:var(--err)">'+E(e.error)+'</td></tr>';
  let u=e.responseHeaders?.['x-usage'];if(u){try{let uj=JSON.parse(u);hd+='<tr><td>Usage</td><td>'+E(JSON.stringify(uj))+'</td></tr>'}catch{}}
  hd+='</tbody></table></div></div>';
  d.innerHTML=tb+cv+rq+rs+hd;
  d.querySelectorAll('.dtab').forEach(t=>t.addEventListener('click',ev=>{ev.stopPropagation();let tn=t.dataset.t;DT[e.id]=tn;d.querySelectorAll('.dtab').forEach(x=>x.classList.toggle('a',x.dataset.t===tn));d.querySelectorAll('.dpan').forEach(x=>x.classList.toggle('a',x.dataset.p===tn))}));
  d.querySelectorAll('[data-i]').forEach(el=>el.addEventListener('click',ev=>{ev.stopPropagation();let i=el.dataset.i,bd=d.querySelector('[data-b="'+i+'"]'),tgs=d.querySelectorAll('[data-i="'+i+'"]');if(bd){bd.classList.toggle('open');let o=bd.classList.contains('open');tgs.forEach(t=>{if(t.classList.contains('ctog'))t.textContent=o?'[collapse]':'[expand]';if(t.classList.contains('ccoll'))t.style.display=o?'none':''})}}))
}
function cpTx(b){let t=b.parentElement.textContent.replace('Copy','').trim();navigator.clipboard.writeText(t).then(()=>{b.textContent='✓';setTimeout(()=>{b.textContent='Copy'},1500)})}
function uSt(){
  document.getElementById('st-tot').textContent=A.length+' request'+(A.length!==1?'s':'');
  let s=A.filter(e=>e.isStreaming&&!e.complete).length,se=document.getElementById('st-str');se.textContent=s+' streaming';se.style.display=s>0?'':'none';
  let er=A.filter(e=>e.error).length,ee=document.getElementById('st-err');ee.textContent=er+' error'+(er!==1?'s':'');ee.style.display=er>0?'':'none'}
document.querySelectorAll('.mbtn').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.mbtn').forEach(x=>x.classList.remove('a'));b.classList.add('a');MF=b.dataset.m;aFilt();render()}));
document.querySelectorAll('.sbtn').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.sbtn').forEach(x=>x.classList.remove('a'));b.classList.add('a');SF=b.dataset.s;aFilt();render()}));
let sT=null;document.getElementById('search').addEventListener('input',e=>{clearTimeout(sT);sT=setTimeout(()=>{SQ=e.target.value;aFilt();render()},150)});
document.getElementById('btn-as').addEventListener('click',()=>{AS=!AS;document.getElementById('btn-as').classList.toggle('btn-a',AS);if(AS){let l=document.getElementById('tlist');l.scrollTop=l.scrollHeight}});
document.getElementById('btn-clr').addEventListener('click',async()=>{await fetch('/api/clear',{method:'POST'})});
document.getElementById('tlist').addEventListener('scroll',e=>{let el=e.target;if(el.scrollHeight-el.scrollTop-el.clientHeight>50&&AS){AS=false;document.getElementById('btn-as').classList.remove('btn-a')}});
document.addEventListener('keydown',e=>{
  if(e.key==='/'&&document.activeElement?.tagName!=='INPUT'){e.preventDefault();document.getElementById('search').focus();return}
  if(document.activeElement?.tagName==='INPUT'){if(e.key==='Escape')document.activeElement.blur();return}
  if(e.key==='ArrowDown'||e.key==='j'){e.preventDefault();SI=Math.min(SI+1,F.length-1);hSel()}
  if(e.key==='ArrowUp'||e.key==='k'){e.preventDefault();SI=Math.max(SI-1,0);hSel()}
  if(e.key==='Enter'&&SI>=0&&SI<F.length){e.preventDefault();tog(F[SI].id)}
  if(e.key==='c'&&SI>=0&&SI<F.length){let en=F[SI];if(!EX.has(en.id))tog(en.id);DT[en.id]='conv';let d=document.getElementById('det-'+en.id);if(d)rDet(d,en)}
});
function hSel(){document.querySelectorAll('.ehdr').forEach((el,i)=>{el.style.outline=i===SI?'1px solid var(--accent)':'';el.style.outlineOffset=i===SI?'-1px':''});if(SI>=0){let es=document.querySelectorAll('.ent');if(es[SI])es[SI].scrollIntoView({block:'nearest'})}}
async function init(){try{let r=await fetch('/api/traffic');A=await r.json();aFilt();render()}catch(e){console.error(e)}connectSSE()}
function connectSSE(){
  let es=new EventSource('/api/events');
  es.addEventListener('connected',()=>{document.getElementById('dot').style.background='var(--success)';document.getElementById('fst').textContent='Connected — live'});
  es.addEventListener('entry',e=>{let en=JSON.parse(e.data);let i=A.findIndex(x=>x.id===en.id);if(i>=0)A[i]=en;else A.push(en);aFilt();render()});
  es.addEventListener('clear',()=>{A=[];EX.clear();DT={};SI=-1;aFilt();render()});
  es.onerror=()=>{document.getElementById('dot').style.background='var(--err)';document.getElementById('fst').textContent='Disconnected — retrying…'};
  es.onopen=()=>{document.getElementById('dot').style.background='var(--success)';document.getElementById('fst').textContent='Connected — live'}}
init();
</script>
</body>
</html>"""
