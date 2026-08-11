/*!
 * @wasmer/sdk
 * Wasmer Javascript SDK. It allows interacting with Wasmer Packages in Node.js and the Browser.
 *
 * @version v0.8.0
 * @author Wasmer Engineering Team <engineering@wasmer.io>
 * @homepage https://github.com/wasmerio/wasmer-js
 * @repository git+https://github.com/wasmerio/wasmer-js.git
 * @license MIT
 */
Error.stackTraceLimit=50,globalThis.onerror=console.error;let a,e=[];globalThis.onmessage=async o=>{if("init"==o.data.type){const{memory:r,module:t,id:i,sdkUrl:l}=o.data,{init:s,ThreadPoolWorker:n}=await import(l);await s({module:t,sdkUrl:l,memory:r}),a=new n(i);for(const o of e.splice(0,e.length))await a.handle(o)}else await(async o=>{a?await a.handle(o):e.push(o)})(o.data)};
