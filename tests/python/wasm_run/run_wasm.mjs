/**
 * Headless runner for fastsim's generated C compiled to WASM, mirroring the
 * in-app browser runner (pathview-fastsim/src/lib/codegen/clang/wasmRunner.ts):
 * same `env` import table (math + libc mem) and same `run()` / `buffer` contract.
 *
 * The test compiles with zig's clang to `wasm32-wasi` (reactor), which statically
 * links libm, so the produced module usually imports nothing; we still pass the
 * production `env` table (harmless if unused) plus a stub for any other declared
 * import, then call the reactor `_initialize` before `run()`. This exercises the
 * real clang->wasm codegen and runs it in a real WASM engine; the `env` import
 * NAMES are covered separately by the import-contract guard in
 * test_codegen_clang_matrix.py.
 *
 * Usage: node run_wasm.mjs <model.wasm> <numStateSignals>
 * Prints JSON {"time":[...], "states":[[...per state...]]} to stdout.
 *
 * KEEP THE env IMPORT TABLE IN SYNC with wasmRunner.ts (WASM_IMPORT_NAMES).
 */
import { readFile } from 'node:fs/promises';

const cRound = (x) => Math.sign(x) * Math.round(Math.abs(x));
const fmin = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.min(a, b));
const fmax = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b));
const copysign = (x, y) => (y < 0 || Object.is(y, -0) ? -1 : 1) * Math.abs(x);

function lgamma(x) {
	if (x <= 0) {
		if (Number.isInteger(x)) return Infinity;
		return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
	}
	const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
		0.1208650973866179e-2, -0.5395239384953e-5];
	let y = x, tmp = x + 5.5;
	tmp -= (x + 0.5) * Math.log(tmp);
	let ser = 1.000000000190015;
	for (let j = 0; j < 6; j++) ser += cof[j] / ++y;
	return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function tgamma(x) {
	if (x > 0) return Math.exp(lgamma(x));
	if (Number.isInteger(x)) return NaN;
	return Math.PI / (Math.sin(Math.PI * x) * Math.exp(lgamma(1 - x)));
}
function erfc(x) {
	const z = Math.abs(x);
	const t = 1 / (1 + 0.5 * z);
	const p = 1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
		t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))));
	const ans = t * Math.exp(-z * z - 1.26551223 + t * p);
	return x >= 0 ? ans : 2 - ans;
}
const erf = (x) => 1 - erfc(x);

const DOUBLE_MATH = {
	sin: Math.sin, cos: Math.cos, tan: Math.tan,
	asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
	sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
	asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
	exp: Math.exp, expm1: Math.expm1,
	log: Math.log, log10: Math.log10, log2: Math.log2, log1p: Math.log1p,
	sqrt: Math.sqrt, cbrt: Math.cbrt, hypot: Math.hypot, pow: Math.pow,
	fabs: Math.abs, floor: Math.floor, ceil: Math.ceil, trunc: Math.trunc, round: cRound,
	fmod: (a, b) => a % b, fmin, fmax,
	fma: (a, b, c) => a * b + c, copysign,
	erf, erfc, lgamma, tgamma
};
const MATH_IMPORTS = Object.fromEntries(
	Object.entries(DOUBLE_MATH).flatMap(([n, f]) => [[n, f], [`${n}f`, f]])
);

function memBuiltins(memory) {
	const u8 = () => new Uint8Array(memory.buffer);
	return {
		memcpy: (d, s, n) => (u8().copyWithin(d, s, s + n), d),
		memmove: (d, s, n) => (u8().copyWithin(d, s, s + n), d),
		memset: (d, c, n) => (u8().fill(c & 0xff, d, d + n), d)
	};
}

const [, , wasmPath, nStateArg] = process.argv;
const numSignals = parseInt(nStateArg, 10);
const mod = await WebAssembly.compile(await readFile(wasmPath));

// A memory may be imported (freestanding) or exported (wasi); cover both.
const memory = new WebAssembly.Memory({ initial: 256, maximum: 65536 });
const env = { memory, ...MATH_IMPORTS, ...memBuiltins(memory) };
const imports = { env };
// Stub any declared import we do not explicitly provide (e.g. unused wasi syscalls).
for (const i of WebAssembly.Module.imports(mod)) {
	if (i.module === 'env' && i.name in env) continue;
	(imports[i.module] ??= {})[i.name] = () => 0;
}
const instance = await WebAssembly.instantiate(mod, imports);
const ex = instance.exports;
if (typeof ex._initialize === 'function') ex._initialize();

const steps = ex.run();
const stride = 1 + numSignals;
const g = ex.buffer;
const ptr = typeof g === 'object' && g !== null ? g.value : (g ?? 0);
const mem = ex.memory ?? memory;
const buf = new Float64Array(mem.buffer, ptr, steps * stride);

const time = new Array(steps);
const states = Array.from({ length: numSignals }, () => new Array(steps));
for (let i = 0; i < steps; i++) {
	time[i] = buf[i * stride];
	for (let s = 0; s < numSignals; s++) states[s][i] = buf[i * stride + 1 + s];
}
process.stdout.write(JSON.stringify({ time, states }));
