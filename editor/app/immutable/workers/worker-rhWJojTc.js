const m="0.29.4",h=`https://cdn.jsdelivr.net/pyodide/v${m}/full/pyodide.mjs`,f=["numpy","scipy","micropip"],d={LOADING_PYODIDE:"Loading Pyodide...",INSTALLING_DEPS:"Installing NumPy and SciPy..."},p={WORKER_NOT_INITIALIZED:"Worker not initialized",FAILED_TO_LOAD_PYODIDE:"Failed to load Pyodide"},y="https://fast.pathsim.org/wheels/fastsim-0.22.0-cp313-cp313-pyemscripten_2025_0_wasm32.whl";y.match(/fastsim-([0-9][^-/]*)-/)?.[1];var g=`"""Shared package-install primitives for the Pyodide and Flask backends.

Single source of truth for "is this importable?" + "install this package",
used by BOTH the engine-install seam (worker boot, via engineInstall.ts) and
the runtime toolbox installer (via TOOLBOX_PYTHON_HELPERS). Defined here, in
one place, so there is exactly one micropip / pip code path with one
error-classification scheme instead of three inline copies.

The engine seam injects this before the worker snapshots \`\`_clean_globals\`\`,
so these names survive a simulation reset; the toolbox layer re-injects the
same source (idempotent) on demand.
"""

import sys as _pv_sys
import importlib as _pv_importlib


def _pv_already_installed(import_path):
    """Return True if the given module path is already importable."""
    if not import_path:
        return False
    try:
        _pv_importlib.import_module(import_path)
        return True
    except Exception:
        return False


async def _pv_install_micropip(spec, pre=False, keep_going=True):
    """Pyodide-side install via micropip (top-level await).

    micropip can only install pure-Python wheels (or packages Pyodide ships
    pre-built), so toolboxes with compiled/native code fail here even though
    they install fine in the standalone (pip-backed) build. On failure we
    classify the error and prefix it with PV_INCOMPATIBLE (browser-runtime
    limitation) or PV_INSTALL_ERROR (genuine failure) so the JS side can show
    a useful hint instead of a raw traceback.

    \`\`pre\`\` allows pre-release wheels (used by the engine seam); \`\`keep_going\`\`
    keeps resolving the rest of the dependency set after a single miss.
    """
    import micropip
    try:
        await micropip.install(spec, keep_going=keep_going, pre=pre)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        incompatible = (
            "pure python" in low
            or "can't find" in low
            or "cannot find" in low
            or "no matching distribution" in low
            or "no known package" in low
        )
        tag = "PV_INCOMPATIBLE" if incompatible else "PV_INSTALL_ERROR"
        raise RuntimeError(tag + ": " + msg)
    return {"ok": True, "spec": spec, "via": "micropip"}


def _pv_install_pip(spec):
    """CPython-side install via subprocess pip (Flask backend)."""
    import subprocess as _pv_subprocess
    res = _pv_subprocess.run(
        [_pv_sys.executable, "-m", "pip", "install", spec],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError("pip install failed:\\n" + (res.stderr or res.stdout))
    return {"ok": True, "spec": spec, "via": "pip"}
`;async function w(t,s){await t.runPythonAsync(g);{await E(t,s);return}}async function E(t,s){s.send({type:"progress",value:"Installing fastsim..."});const e=new URL(y,self.location.origin).href;await t.runPythonAsync(`await _pv_install_micropip(${JSON.stringify(e)})`),await t.runPythonAsync(`
import fastsim
print(f"fastsim {fastsim.__version__} loaded successfully")
	`)}let r=null,u=!1,c=!1;const l=[];function n(t){postMessage(t)}async function b(t){if(u){n({type:"ready"});return}n({type:"progress",value:d.LOADING_PYODIDE});const{loadPyodide:s}=await import(h);if(r=await s(),!r)throw new Error(p.FAILED_TO_LOAD_PYODIDE);r.setStdout({batched:e=>n({type:"stdout",value:e})}),r.setStderr({batched:e=>n({type:"stderr",value:e})}),n({type:"progress",value:d.INSTALLING_DEPS}),await r.loadPackage([...f]),await w(r,{send:n,token:t}),await r.runPythonAsync("import numpy as np"),await r.runPythonAsync("import gc"),await r.runPythonAsync("_clean_globals = set(globals().keys())"),u=!0,n({type:"ready"})}async function I(t,s){if(!r)throw new Error(p.WORKER_NOT_INITIALIZED);try{await r.runPythonAsync(s),n({type:"ok",id:t})}catch(e){const i=e instanceof Error?e.message:String(e);let a;try{a=await r.runPythonAsync(`
import traceback
traceback.format_exc()
			`)}catch{}n({type:"error",id:t,error:i,traceback:a})}}async function v(t,s){if(!r)throw new Error(p.WORKER_NOT_INITIALIZED);try{const e=await r.runPythonAsync(`
_eval_result = ${s}
json.dumps(_eval_result, default=_to_json if '_to_json' in dir() else str)
		`);n({type:"value",id:t,value:e})}catch(e){const i=e instanceof Error?e.message:String(e);let a;try{a=await r.runPythonAsync(`
import traceback
traceback.format_exc()
			`)}catch{}n({type:"error",id:t,error:i,traceback:a})}}async function k(t,s){if(!r)throw new Error(p.WORKER_NOT_INITIALIZED);c=!0,l.length=0;try{for(;c;){for(;l.length>0;){const a=l.shift();try{await r.runPythonAsync(a)}catch(o){const _=o instanceof Error?o.message:String(o);n({type:"stderr",value:`Stream exec error: ${_}`})}}const e=await r.runPythonAsync(`
_eval_result = ${s}
json.dumps(_eval_result, default=_to_json if '_to_json' in dir() else str)
			`),i=JSON.parse(e);if(!c){!i.done&&i.result&&n({type:"stream-data",id:t,value:e});break}if(i.done)break;n({type:"stream-data",id:t,value:e})}}catch(e){const i=e instanceof Error?e.message:String(e);let a;try{a=await r.runPythonAsync(`
import traceback
traceback.format_exc()
			`)}catch{}n({type:"error",id:t,error:i,traceback:a})}finally{c=!1,n({type:"stream-done",id:t})}}function P(){c=!1}self.onmessage=async t=>{const{type:s}=t.data,e="id"in t.data?t.data.id:void 0,i="code"in t.data?t.data.code:void 0,a="expr"in t.data?t.data.expr:void 0;try{switch(s){case"init":await b("token"in t.data?t.data.token:void 0);break;case"exec":if(!e||typeof i!="string")throw new Error("Invalid exec request: missing id or code");await I(e,i);break;case"eval":if(!e||typeof a!="string")throw new Error("Invalid eval request: missing id or expr");await v(e,a);break;case"stream-start":if(!e||typeof a!="string")throw new Error("Invalid stream-start request: missing id or expr");k(e,a);break;case"stream-stop":P();break;case"stream-exec":typeof i=="string"&&c&&l.push(i);break;default:throw new Error(`Unknown message type: ${s}`)}}catch(o){n({type:"error",id:e,error:o instanceof Error?o.message:String(o)})}};
