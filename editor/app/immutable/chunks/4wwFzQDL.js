import{bc as Os,n as zs,A as Ns,U as cs,aA as Hs,h as Es,bd as Zs,ba as Xs,p as os,i as U,j as B,t as L,g as l,u as q,r as Z,e as rs,f as G,F as fs,ag as Ys,d as ds,s as hs,m as Ds}from"./Cb15qHYK.js";import{B as Vs,s as r,b as Js}from"./BND8GRlV.js";import{o as I,a as g,c as X,s as Rs,f as Bs}from"./D7NvfH0d.js";import{p as K,i as H}from"./DymO2F91.js";import{e as as,i as ns,a as js}from"./p_yRNlWA.js";import{_ as Qs,b as Ts}from"./B23gN1co.js";function me(a,n,s=n){var e=new WeakSet;Os(a,"input",async t=>{var i=t?a.defaultValue:a.value;if(i=ys(a)?ks(i):i,s(i),cs!==null&&e.add(cs),await Hs(),i!==(i=n())){var p=a.selectionStart,o=a.selectionEnd,c=a.value.length;if(a.value=i??"",o!==null){var u=a.value.length;p===o&&o===c&&u>c?(a.selectionStart=u,a.selectionEnd=u):(a.selectionStart=p,a.selectionEnd=Math.min(o,u))}}}),(Es&&a.defaultValue!==a.value||zs(n)==null&&a.value)&&(s(ys(a)?ks(a.value):a.value),cs!==null&&e.add(cs)),Ns(()=>{var t=n();if(a===document.activeElement){var i=Zs??cs;if(e.has(i))return}ys(a)&&t===ks(a.value)||a.type==="date"&&!t&&!a.value||t!==a.value&&(a.value=t??"")})}function fe(a,n,s=n){Os(a,"change",e=>{var t=e?a.defaultChecked:a.checked;s(t)}),(Es&&a.defaultChecked!==a.checked||zs(n)==null)&&s(a.checked),Ns(()=>{var e=n();a.checked=!!e})}function ys(a){var n=a.type;return n==="number"||n==="range"}function ks(a){return a===""?null:+a}const $s=Vs.accent,qs={default:"#969696"},he=[$s,"#E57373","#FFB74D","#FFF176","#81C784","#4DB6AC","#4DD0E1","#64B5F6","#BA68C8","#F06292","#90A4AE","#FFFFFF"];function sa(a){const{name:n,category:s,description:e=`${n} block`,blockClass:t,importPath:i,inputs:p=["in 0"],outputs:o=["out 0"],minInputs:c=1,minOutputs:u=1,maxInputs:d=null,maxOutputs:v=null,syncPorts:A,shape:P,params:O={}}=a,S=Object.entries(O).map(([C,w])=>({name:C,type:w.type,default:w.default,description:w.description,min:w.min,max:w.max,options:w.options}));return{type:t,name:n,category:s,description:e,blockClass:t,importPath:i,shape:P,ports:{inputs:p.map(C=>({name:C,direction:"input",color:qs.default})),outputs:o.map(C=>({name:C,direction:"output",color:qs.default})),minInputs:c,minOutputs:u,maxInputs:d,maxOutputs:v,syncPorts:A},params:S}}const aa={Constant:{blockClass:"Constant",description:"Produces a constant output signal (SISO).",docstringHtml:`<p>Produces a constant output signal (SISO).</p>
<div class="math">
\\begin{equation*}
y(t) = const.
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>value <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>constant defining block output</dd>
</dl>
</div>
`,params:{value:{type:"any",default:null,description:"constant defining block output"}},inputs:[],outputs:["out"]},Source:{blockClass:"Source",description:"Source that produces an arbitrary time dependent output defined by `func` (callable).",docstringHtml:`<p>Source that produces an arbitrary time dependent output defined by <cite>func</cite> (callable).</p>
<div class="math">
\\begin{equation*}
y(t) = \\mathrm{func}(t)
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its internal function (<cite>func</cite>) will
be called multiple times per timestep, each time when <cite>Simulation._update(t)</cite>
is called in the global simulation loop.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>For example a ramp:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Source</span><span class="whitespace">

</span><span class="name">src</span> <span class="operator">=</span> <span class="name">Source</span><span class="punctuation">(</span><span class="keyword">lambda</span> <span class="name">t</span> <span class="punctuation">:</span> <span class="name">t</span><span class="punctuation">)</span>
</pre>
<p>or a simple sinusoid with some frequency:</p>
<pre class="code python literal-block">
<span class="keyword namespace">import</span> <span class="name namespace">numpy</span> <span class="keyword">as</span> <span class="name namespace">np</span><span class="whitespace">
</span><span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Source</span><span class="whitespace">

</span><span class="comment single">#some parameter</span><span class="whitespace">
</span><span class="name">omega</span> <span class="operator">=</span> <span class="literal number integer">100</span><span class="whitespace">

</span><span class="comment single">#the function that gets evaluated</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">f</span><span class="punctuation">(</span><span class="name">t</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">np</span><span class="operator">.</span><span class="name">sin</span><span class="punctuation">(</span><span class="name">omega</span> <span class="operator">*</span> <span class="name">t</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="name">src</span> <span class="operator">=</span> <span class="name">Source</span><span class="punctuation">(</span><span class="name">f</span><span class="punctuation">)</span>
</pre>
<p>Because the <cite>Source</cite> block only has a single argument, it can be
used to decorate a function and make it a <cite>PathSim</cite> block. This might
be handy in some cases to keep definitions concise and localized
in the code:</p>
<pre class="code python literal-block">
<span class="keyword namespace">import</span> <span class="name namespace">numpy</span> <span class="keyword">as</span> <span class="name namespace">np</span><span class="whitespace">
</span><span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Source</span><span class="whitespace">

</span><span class="comment single">#does the same as the definition above</span><span class="whitespace">

</span><span class="name decorator">&#64;Source</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">src</span><span class="punctuation">(</span><span class="name">t</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="name">omega</span> <span class="operator">=</span> <span class="literal number integer">100</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">np</span><span class="operator">.</span><span class="name">sin</span><span class="punctuation">(</span><span class="name">omega</span> <span class="operator">*</span> <span class="name">t</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="comment single">#'src' is now a PathSim block</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>function defining time dependent block output</dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"function defining time dependent block output"}},inputs:[],outputs:["out"]},SinusoidalSource:{blockClass:"SinusoidalSource",description:"Source block that generates a sinusoid wave",docstringHtml:`<p>Source block that generates a sinusoid wave</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>frequency <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>frequency of the sinusoid</dd>
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplitude of the sinusoid</dd>
<dt>phase <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>phase of the sinusoid</dd>
</dl>
</div>
`,params:{frequency:{type:"any",default:null,description:"frequency of the sinusoid"},amplitude:{type:"any",default:null,description:"amplitude of the sinusoid"},phase:{type:"any",default:null,description:"phase of the sinusoid"}},inputs:[],outputs:["out"]},StepSource:{blockClass:"StepSource",description:"Discrete time unit step (or multi step) source block.",docstringHtml:`<p>Discrete time unit step (or multi step) source block.</p>
<p>Utilizes a scheduled event to set the block output
to the specified output levels at the defined event times.</p>
<p>The arguments can be vectorial and in that case, the output is set to the
amplitude that corresponds to the defined delay like a zero-order-hold stage.
This functionality enables adding external or time series measurement data
into the system.</p>
<div class="section" id="examples">
<h3>Examples</h3>
<p>This is how to use the source as a unit step source:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">StepSource</span><span class="whitespace">

</span><span class="comment single">#default, starts at 0, jumps to 1</span><span class="whitespace">
</span><span class="name">stp</span> <span class="operator">=</span> <span class="name">StepSource</span><span class="punctuation">()</span>
</pre>
<p>And this is how to configure it with multiple consecutive steps:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">StepSource</span><span class="whitespace">

</span><span class="comment single">#starts at 0, jumps to 1 at 1, jumps to -1 at 2 and jumps back to 0 at 3</span><span class="whitespace">
</span><span class="name">stp</span> <span class="operator">=</span> <span class="name">StepSource</span><span class="punctuation">(</span><span class="name">amplitude</span><span class="operator">=</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">,</span> <span class="operator">-</span><span class="literal number integer">1</span><span class="punctuation">,</span> <span class="literal number integer">0</span><span class="punctuation">],</span> <span class="name">tau</span><span class="operator">=</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">,</span> <span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">3</span><span class="punctuation">])</span>
</pre>
<p>Similarly implementing measured time series data via zoh:</p>
<pre class="code python literal-block">
<span class="keyword namespace">import</span> <span class="name namespace">numpy</span> <span class="keyword">as</span> <span class="name namespace">np</span><span class="whitespace">
</span><span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">StepSource</span><span class="whitespace">

</span><span class="comment single">#some random time series arrays</span><span class="whitespace">
</span><span class="name">times</span><span class="punctuation">,</span> <span class="name">data</span> <span class="operator">=</span> <span class="name">np</span><span class="operator">.</span><span class="name">linspace</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">,</span> <span class="literal number integer">100</span><span class="punctuation">,</span> <span class="literal number integer">1000</span><span class="punctuation">),</span> <span class="name">np</span><span class="operator">.</span><span class="name">random</span><span class="operator">.</span><span class="name">rand</span><span class="punctuation">(</span><span class="literal number integer">1000</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="comment single">#pass them to the block</span><span class="whitespace">
</span><span class="name">stp</span> <span class="operator">=</span> <span class="name">StepSource</span><span class="punctuation">(</span><span class="name">amplitude</span><span class="operator">=</span><span class="name">data</span><span class="punctuation">,</span> <span class="name">tau</span><span class="operator">=</span><span class="name">times</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float | list[float]</span></dt>
<dd>amplitude of the step signal, or amplitudes / output
levels of the multiple steps</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float | list[float]</span></dt>
<dd>delay of the step, or delays of the different steps</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>Evt <span class="classifier-delimiter">:</span> <span class="classifier">ScheduleList</span></dt>
<dd>internal scheduled event directly accessible</dd>
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[ScheduleList]</span></dt>
<dd>list of interna events</dd>
</dl>
</div>
`,params:{amplitude:{type:"any",default:null,description:"amplitude of the step signal, or amplitudes / output levels of the multiple steps"},tau:{type:"any",default:null,description:"delay of the step, or delays of the different steps"}},inputs:[],outputs:["out"]},PulseSource:{blockClass:"PulseSource",description:"Generates a periodic pulse waveform with defined rise and fall times.",docstringHtml:`<p>Generates a periodic pulse waveform with defined rise and fall times.</p>
<p>Scheduled events trigger phase changes (low, rising, high, falling),
and the <cite>update</cite> method calculates the output value based on the
current phase, performing linear interpolation during rise and fall.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Peak amplitude of the pulse. Default is 1.0.</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Period of the pulse train. Must be positive. Default is 1.0.</dd>
<dt>t_rise <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Duration of the rising edge. Default is 0.0.</dd>
<dt>t_fall <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Duration of the falling edge. Default is 0.0.</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Initial delay before the first pulse cycle begins. Default is 0.0.</dd>
<dt>duty <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Duty cycle, ratio of the pulse ON duration (plateau time only)
to the total period T (must be between 0 and 1). Default is 0.5.
The high plateau duration is <cite>T * duty</cite>.</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>Internal scheduled events triggering phase transitions.</dd>
<dt>_phase <span class="classifier-delimiter">:</span> <span class="classifier">str</span></dt>
<dd>Current phase of the pulse ('low', 'rising', 'high', 'falling').</dd>
<dt>_phase_start_time <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>Simulation time when the current phase began.</dd>
</dl>
</div>
`,params:{amplitude:{type:"number",default:"1.0",description:"Peak amplitude of the pulse. Default is 1.0."},T:{type:"number",default:"1.0",description:"Period of the pulse train. Must be positive. Default is 1.0."},t_rise:{type:"number",default:"0.0",description:"Duration of the rising edge. Default is 0.0."},t_fall:{type:"number",default:"0.0",description:"Duration of the falling edge. Default is 0.0."},tau:{type:"number",default:"0.0",description:"Initial delay before the first pulse cycle begins. Default is 0.0."},duty:{type:"number",default:"0.5",description:"Duty cycle, ratio of the pulse ON duration (plateau time only) to the total period T (must be between 0 and 1). Default is 0.5. The high plateau duration is `T * duty`."}},inputs:[],outputs:["out"]},TriangleWaveSource:{blockClass:"TriangleWaveSource",description:"Source block that generates an analog triangle wave",docstringHtml:`<p>Source block that generates an analog triangle wave</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>frequency <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>frequency of the triangle wave</dd>
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplitude of the triangle wave</dd>
<dt>phase <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>phase of the triangle wave</dd>
</dl>
</div>
`,params:{frequency:{type:"any",default:null,description:"frequency of the triangle wave"},amplitude:{type:"any",default:null,description:"amplitude of the triangle wave"},phase:{type:"any",default:null,description:"phase of the triangle wave"}},inputs:[],outputs:["out"]},SquareWaveSource:{blockClass:"SquareWaveSource",description:"Discrete time square wave source.",docstringHtml:`<p>Discrete time square wave source.</p>
<p>Utilizes scheduled events to periodically set
the block output at discrete times.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplitude of the square wave signal</dd>
<dt>frequency <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>frequency of the square wave signal</dd>
<dt>phase <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>phase of the square wave signal</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled events</dd>
</dl>
</div>
`,params:{amplitude:{type:"any",default:null,description:"amplitude of the square wave signal"},frequency:{type:"any",default:null,description:"frequency of the square wave signal"},phase:{type:"any",default:null,description:"phase of the square wave signal"}},inputs:[],outputs:["out"]},GaussianPulseSource:{blockClass:"GaussianPulseSource",description:"Source block that generates a gaussian pulse",docstringHtml:`<p>Source block that generates a gaussian pulse</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplitude of the gaussian pulse</dd>
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>maximum frequency component of the gaussian pulse (steepness)</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>time delay of the gaussian pulse</dd>
</dl>
</div>
`,params:{amplitude:{type:"any",default:null,description:"amplitude of the gaussian pulse"},f_max:{type:"any",default:null,description:"maximum frequency component of the gaussian pulse (steepness)"},tau:{type:"any",default:null,description:"time delay of the gaussian pulse"}},inputs:[],outputs:["out"]},ChirpPhaseNoiseSource:{blockClass:"ChirpPhaseNoiseSource",description:"Chirp source, sinusoid with frequency ramp up and ramp down, plus phase noise.",docstringHtml:`<p>Chirp source, sinusoid with frequency ramp up and ramp down, plus phase noise.</p>
<p>This works by using a time dependent triangle wave for the frequency
and integrating it with a numerical integration engine to get a
continuous phase. This phase is then used to evaluate a sinusoid.</p>
<p>Additionally the chirp source can have white and cumulative phase noise.
Mathematically it looks like this for the contributions to the phase from
the triangular wave:</p>
<div class="math">
\\begin{equation*}
\\varphi_t(t) = \\int_0^t \\mathrm{tri}_{f_0, B, T}(\\tau) \\, d\\tau
\\end{equation*}
</div>
<p>And from the white (w) and cumulative (c) noise:</p>
<div class="math">
\\begin{equation*}
\\varphi_n(t) = \\sigma_w \\, n_w(t) + \\sigma_c \\int_0^t n_c(\\tau) \\, d\\tau
\\end{equation*}
</div>
<p>The phase contributions are then used to evaluate a sinusoid to get the final chirp signal:</p>
<div class="math">
\\begin{equation*}
y(t) = A \\sin(\\varphi_t(t) + \\varphi_n(t) + \\varphi_0)
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>amplitude <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplitude of the chirp signal</dd>
<dt>f0 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>start frequency of the chirp signal</dd>
<dt>BW <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>bandwidth of the frequency ramp of the chirp signal</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>period of the frequency ramp of the chirp signal</dd>
<dt>phase <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>phase of sinusoid (initial, radians)</dd>
<dt>sig_cum <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>weight for cumulative phase noise contribution</dd>
<dt>sig_white <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>weight for white phase noise contribution</dd>
<dt>sampling_period <span class="classifier-delimiter">:</span> <span class="classifier">float, None</span></dt>
<dd>time between phase noise samples. If None,
noise is sampled every timestep (default is 0.1)</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>noise_1 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>internal noise value for white phase noise</dd>
<dt>noise_2 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>internal noise value for cumulative phase noise</dd>
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>scheduled event for periodic sampling (only if sampling_period is set)</dd>
</dl>
</div>
`,params:{amplitude:{type:"number",default:"1.0",description:"amplitude of the chirp signal"},f0:{type:"number",default:"1.0",description:"start frequency of the chirp signal"},BW:{type:"number",default:"1.0",description:"bandwidth of the frequency ramp of the chirp signal"},T:{type:"number",default:"1.0",description:"period of the frequency ramp of the chirp signal"},phase:{type:"number",default:"0.0",description:"phase of sinusoid (initial, radians)"},sig_cum:{type:"number",default:"0.0",description:"weight for cumulative phase noise contribution"},sig_white:{type:"number",default:"0.0",description:"weight for white phase noise contribution"},sampling_period:{type:"number",default:"0.1",description:"time between phase noise samples. If None, noise is sampled every timestep (default is 0.1)"},seed:{type:"any",default:null,description:""}},inputs:[],outputs:["out"]},ClockSource:{blockClass:"ClockSource",description:"Discrete time clock source block.",docstringHtml:`<p>Discrete time clock source block.</p>
<p>Utilizes scheduled events to periodically set
the block output to 0 or 1 at discrete times.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>period of the clock</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>clock delay</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event list</dd>
</dl>
</div>
`,params:{T:{type:"number",default:"1.0",description:"period of the clock"},tau:{type:"number",default:"0.0",description:"clock delay"}},inputs:[],outputs:["out"]},WhiteNoise:{blockClass:"WhiteNoise",description:"White noise source with Gaussian distribution.",docstringHtml:`<p>White noise source with Gaussian distribution.</p>
<p>Generates uncorrelated random samples with either constant amplitude
(<tt class="docutils literal">standard_deviation</tt> mode) or timestep-scaled amplitude for stochastic
integration (<tt class="docutils literal">spectral_density</tt> mode).</p>
<p>In spectral density mode, output is scaled as √(S₀/dt) so that integrating
the noise yields correct statistical properties (Wiener process).</p>
<div class="section" id="note">
<h3>Note</h3>
<p>If <tt class="docutils literal">spectral_density</tt> is provided, it takes precedence over <tt class="docutils literal">standard_deviation</tt>.
If <tt class="docutils literal">sampling_period</tt> is set, noise is sampled at fixed intervals (zero-order hold).</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>standard_deviation <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>output standard deviation for constant-amplitude mode (default: 1.0)</dd>
<dt>spectral_density <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>power spectral density S₀ in [signal²/Hz]</dd>
<dt>sampling_period <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>time between samples, if None samples every timestep</dd>
<dt>seed <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span></dt>
<dd>random seed for reproducibility</dd>
</dl>
</div>
`,params:{standard_deviation:{type:"number",default:"1.0",description:"output standard deviation for constant-amplitude mode (default: 1.0)"},spectral_density:{type:"any",default:null,description:"power spectral density S₀ in [signal²/Hz]"},sampling_period:{type:"any",default:null,description:"time between samples, if None samples every timestep"},seed:{type:"any",default:null,description:"random seed for reproducibility"}},inputs:[],outputs:["out"]},PinkNoise:{blockClass:"PinkNoise",description:"Pink noise (1/f noise) source using the Voss-McCartney algorithm.",docstringHtml:`<p>Pink noise (1/f noise) source using the Voss-McCartney algorithm.</p>
<p>Generates noise with power spectral density proportional to 1/f, where
lower frequencies have more power than higher frequencies.</p>
<p>The algorithm maintains <tt class="docutils literal">num_octaves</tt> independent random values representing
different frequency bands. At each sample, one octave is updated based on the
binary representation of the sample counter, creating the characteristic 1/f
spectrum through the superposition of different update rates.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>If <tt class="docutils literal">spectral_density</tt> is provided, it takes precedence over <tt class="docutils literal">standard_deviation</tt>.
If <tt class="docutils literal">sampling_period</tt> is set, noise is sampled at fixed intervals (zero-order hold).</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>standard_deviation <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>approximate output standard deviation (default: 1.0)</dd>
<dt>spectral_density <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>power spectral density, output scaled as √(S₀/(N·dt))</dd>
<dt>num_octaves <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>number of frequency bands in algorithm (default: 16)</dd>
<dt>sampling_period <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>time between samples, if None samples every timestep</dd>
<dt>seed <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span></dt>
<dd>random seed for reproducibility</dd>
</dl>
</div>
`,params:{standard_deviation:{type:"number",default:"1.0",description:"approximate output standard deviation (default: 1.0)"},spectral_density:{type:"any",default:null,description:"power spectral density, output scaled as √(S₀/(N·dt))"},num_octaves:{type:"integer",default:"16",description:"number of frequency bands in algorithm (default: 16)"},sampling_period:{type:"any",default:null,description:"time between samples, if None samples every timestep"},seed:{type:"any",default:null,description:"random seed for reproducibility"}},inputs:[],outputs:["out"]},RandomNumberGenerator:{blockClass:"RandomNumberGenerator",description:"Generates a random output value using `numpy.random.rand`.",docstringHtml:`<p>Generates a random output value using <cite>numpy.random.rand</cite>.</p>
<p>If no <cite>sampling_period</cite> (None) is specified, every simulation timestep gets
a random value. Otherwise an internal <cite>Schedule</cite> event is used to periodically
sample a random value and set the output like a zero-order-hold stage.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>sampling_period <span class="classifier-delimiter">:</span> <span class="classifier">float, None</span></dt>
<dd>time between random samples</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>_sample <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>internal random number state in case that
no <cite>sampling_period</cite> is provided</dd>
<dt>Evt <span class="classifier-delimiter">:</span> <span class="classifier">Schedule</span></dt>
<dd>internal event that periodically samples a random
value in case <cite>sampling_period</cite> is provided</dd>
</dl>
</div>
`,params:{sampling_period:{type:"any",default:null,description:"time between random samples"},seed:{type:"any",default:null,description:""}},inputs:[],outputs:["out"]},Integrator:{blockClass:"Integrator",description:"Integrates the input signal.",docstringHtml:`<p>Integrates the input signal.</p>
<p>Uses a numerical integration engine like this:</p>
<div class="math">
\\begin{equation*}
y(t) = \\int_0^t u(\\tau) \\ d \\tau
\\end{equation*}
</div>
<p>or in differential form like this:</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x}(t) &amp;= u(t) \\\\
           y(t) &amp;= x(t)
\\end{align}
\\end{equation*}
</div>
<p>The Integrator block is inherently MIMO capable, so <cite>u</cite>
and <cite>y</cite> can be vectors.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>This is how to initialize the integrator:</p>
<pre class="code python literal-block">
<span class="comment single">#initial value 0.0</span><span class="whitespace">
</span><span class="name">i1</span> <span class="operator">=</span> <span class="name">Integrator</span><span class="punctuation">()</span><span class="whitespace">

</span><span class="comment single">#initial value 2.5</span><span class="whitespace">
</span><span class="name">i2</span> <span class="operator">=</span> <span class="name">Integrator</span><span class="punctuation">(</span><span class="literal number float">2.5</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">float, array</span></dt>
<dd>initial value of integrator</dd>
</dl>
</div>
`,params:{initial_value:{type:"any",default:null,description:"initial value of integrator"}},inputs:null,outputs:null},Differentiator:{blockClass:"Differentiator",description:"Differentiates the input signal.",docstringHtml:`<p>Differentiates the input signal.</p>
<p>Uses a first order transfer function with a pole at the origin which implements
a high pass filter. Supports vector input.</p>
<div class="math">
\\begin{equation*}
H_\\mathrm{diff}(s) = \\frac{s}{1 + s / f_\\mathrm{max}}
\\end{equation*}
</div>
<p>The approximation holds for signals up to a frequency of approximately f_max.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>Depending on <cite>f_max</cite>, the resulting system might become stiff or ill conditioned!
As a practical choice set <cite>f_max</cite> to 3x the highest expected signal frequency.</p>
</div>
<div class="section" id="note-1">
<h3>Note</h3>
<p>Since this is an approximation of real differentiation, the approximation will not hold
if there are high frequency components present in the signal. For example if you have
discontinuities such as steps or squere waves.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#cutoff at 1kHz</span><span class="whitespace">
</span><span class="name">D</span> <span class="operator">=</span> <span class="name">Differentiator</span><span class="punctuation">(</span><span class="name">f_max</span><span class="operator">=</span><span class="literal number float">1e3</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>highest expected signal frequency</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_dyn <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal dynamic operator for ODE component</dd>
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{f_max:{type:"any",default:null,description:"highest expected signal frequency"}},inputs:null,outputs:null},Delay:{blockClass:"Delay",description:"Delays the input signal by a time constant 'tau' in seconds.",docstringHtml:`<p>Delays the input signal by a time constant 'tau' in seconds.</p>
<p>Supports two modes of operation:</p>
<p><strong>Continuous mode</strong> (default, <tt class="docutils literal">sampling_period=None</tt>):
Uses an adaptive interpolating buffer for continuous-time delay.</p>
<div class="math">
\\begin{equation*}
y(t) =
\\begin{cases}
x(t - \\tau) &amp; , t \\geq \\tau \\\\
0            &amp; , t &lt; \\tau
\\end{cases}
\\end{equation*}
</div>
<p><strong>Discrete mode</strong> (<tt class="docutils literal">sampling_period</tt> provided):
Uses a ring buffer with scheduled sampling events for N-sample delay,
where <tt class="docutils literal">N = round(tau / sampling_period)</tt>.</p>
<div class="math">
\\begin{equation*}
y[k] = x[k - N]
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>In continuous mode, the internal adaptive buffer uses interpolation for
the evaluation. This is required to be compatible with variable step solvers.
It has a drawback however. The order of the ode solver used will degrade
when this block is used, due to the interpolation.</p>
</div>
<div class="section" id="note-1">
<h3>Note</h3>
<p>This block supports vector input, meaning we can have multiple parallel
delay paths through this block.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>Continuous-time delay:</p>
<pre class="code python literal-block">
<span class="comment single">#5 time units delay</span><span class="whitespace">
</span><span class="name">D</span> <span class="operator">=</span> <span class="name">Delay</span><span class="punctuation">(</span><span class="name">tau</span><span class="operator">=</span><span class="literal number integer">5</span><span class="punctuation">)</span>
</pre>
<p>Discrete-time N-sample delay (10 samples):</p>
<pre class="code python literal-block">
<span class="name">D</span> <span class="operator">=</span> <span class="name">Delay</span><span class="punctuation">(</span><span class="name">tau</span><span class="operator">=</span><span class="literal number float">0.01</span><span class="punctuation">,</span> <span class="name">sampling_period</span><span class="operator">=</span><span class="literal number float">0.001</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay time constant in seconds</dd>
<dt>sampling_period <span class="classifier-delimiter">:</span> <span class="classifier">float, None</span></dt>
<dd>sampling period for discrete mode, default is continuous mode</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>_buffer <span class="classifier-delimiter">:</span> <span class="classifier">AdaptiveBuffer</span></dt>
<dd>internal interpolatable adaptive rolling buffer (continuous mode)</dd>
<dt>_ring <span class="classifier-delimiter">:</span> <span class="classifier">deque</span></dt>
<dd>internal ring buffer for N-sample delay (discrete mode)</dd>
</dl>
</div>
`,params:{tau:{type:"number",default:"0.001",description:"delay time constant in seconds"},sampling_period:{type:"any",default:null,description:"sampling period for discrete mode, default is continuous mode"}},inputs:null,outputs:null},ODE:{blockClass:"ODE",description:"Ordinary differential equation (ODE) defined by its right hand side function.",docstringHtml:`<p>Ordinary differential equation (ODE) defined by its right hand side function.</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x}(t) &amp;= \\mathrm{func}(x(t), u(t), t) \\\\
           y(t) &amp;= x(t)
\\end{align}
\\end{equation*}
</div>
<p>with inhomogenity (input) <cite>u</cite> and state vector <cite>x</cite>. The function can be nonlinear
and the ODE can be of arbitrary order. The block utilizes the integration engine
to solve the ODE by integrating the <cite>func</cite>, which is the right hand side function.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>For example a linear 1st order ODE:</p>
<pre class="code python literal-block">
<span class="name">ode</span> <span class="operator">=</span> <span class="name">ODE</span><span class="punctuation">(</span><span class="keyword">lambda</span> <span class="name">x</span><span class="punctuation">,</span> <span class="name">u</span><span class="punctuation">,</span> <span class="name">t</span><span class="punctuation">:</span> <span class="operator">-</span><span class="name">x</span><span class="punctuation">)</span>
</pre>
<p>Or something more complex like the <cite>Van der Pol</cite> system, where it makes sense to
also specify the jacobian, which improves convergence for implicit solvers but is
not needed in most cases:</p>
<pre class="code python literal-block">
<span class="keyword namespace">import</span> <span class="name namespace">numpy</span> <span class="keyword">as</span> <span class="name namespace">np</span><span class="whitespace">

</span><span class="comment single">#initial condition</span><span class="whitespace">
</span><span class="name">x0</span> <span class="operator">=</span> <span class="name">np</span><span class="operator">.</span><span class="name">array</span><span class="punctuation">([</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">0</span><span class="punctuation">])</span><span class="whitespace">

</span><span class="comment single">#van der Pol parameter</span><span class="whitespace">
</span><span class="name">mu</span> <span class="operator">=</span> <span class="literal number integer">1000</span><span class="whitespace">

</span><span class="keyword">def</span> <span class="name function">func</span><span class="punctuation">(</span><span class="name">x</span><span class="punctuation">,</span> <span class="name">u</span><span class="punctuation">,</span> <span class="name">t</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">np</span><span class="operator">.</span><span class="name">array</span><span class="punctuation">([</span><span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">],</span> <span class="name">mu</span><span class="operator">*</span><span class="punctuation">(</span><span class="literal number integer">1</span> <span class="operator">-</span> <span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">0</span><span class="punctuation">]</span><span class="operator">**</span><span class="literal number integer">2</span><span class="punctuation">)</span><span class="operator">*</span><span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">]</span> <span class="operator">-</span> <span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">0</span><span class="punctuation">]])</span><span class="whitespace">

</span><span class="comment single">#analytical jacobian (optional)</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">jac</span><span class="punctuation">(</span><span class="name">x</span><span class="punctuation">,</span> <span class="name">u</span><span class="punctuation">,</span> <span class="name">t</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">np</span><span class="operator">.</span><span class="name">array</span><span class="punctuation">(</span><span class="whitespace">
</span>        <span class="punctuation">[[</span><span class="literal number integer">0</span>                <span class="punctuation">,</span> <span class="literal number integer">1</span>               <span class="punctuation">],</span><span class="whitespace">
</span>         <span class="punctuation">[</span><span class="operator">-</span><span class="name">mu</span><span class="operator">*</span><span class="literal number integer">2</span><span class="operator">*</span><span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">0</span><span class="punctuation">]</span><span class="operator">*</span><span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">]</span><span class="operator">-</span><span class="literal number integer">1</span><span class="punctuation">,</span> <span class="name">mu</span><span class="operator">*</span><span class="punctuation">(</span><span class="literal number integer">1</span> <span class="operator">-</span> <span class="name">x</span><span class="punctuation">[</span><span class="literal number integer">0</span><span class="punctuation">]</span><span class="operator">**</span><span class="literal number integer">2</span><span class="punctuation">)]]</span><span class="whitespace">
</span>         <span class="punctuation">)</span><span class="whitespace">

</span><span class="comment single">#finally the block</span><span class="whitespace">
</span><span class="name">vdp</span> <span class="operator">=</span> <span class="name">ODE</span><span class="punctuation">(</span><span class="name">func</span><span class="punctuation">,</span> <span class="name">x0</span><span class="punctuation">,</span> <span class="name">jac</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>right hand side function of ODE</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array[float]</span></dt>
<dd>initial state / initial condition</dd>
<dt>jac <span class="classifier-delimiter">:</span> <span class="classifier">callable, None</span></dt>
<dd>jacobian of 'func' or 'None'</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_dyn <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal dynamic operator for ODE right hand side 'func'</dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"right hand side function of ODE"},initial_value:{type:"any",default:null,description:"initial state / initial condition"},jac:{type:"any",default:null,description:"jacobian of 'func' or 'None'"}},inputs:null,outputs:null},DynamicalSystem:{blockClass:"DynamicalSystem",description:"This block implements a nonlinear dynamical system / nonlinear state space model.",docstringHtml:`<p>This block implements a nonlinear dynamical system / nonlinear state space model.</p>
<p>Its basically the same as the <cite>ODE</cite> block with the addition of an output equation
that takes the state, input and time as arguments:</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x}(t) &amp;= \\mathrm{func}_\\mathrm{dyn}(x(t), u(t), t) \\\\
           y(t) &amp;= \\mathrm{func}_\\mathrm{alg}(x(t), u(t), t)
\\end{align}
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func_dyn <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>right hand side function of ode-part of the system</dd>
<dt>func_alg <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>output function of the system</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array[float]</span></dt>
<dd>initial state / initial condition</dd>
<dt>jac_dyn <span class="classifier-delimiter">:</span> <span class="classifier">callable | None</span></dt>
<dd>optional jacobian of <cite>func_dyn</cite> to improve convergence
for implicit ode solvers</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_dyn <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal dynamic operator for <cite>func_dyn</cite></dd>
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal dynamic operator for <cite>func_alg</cite></dd>
</dl>
</div>
`,params:{func_dyn:{type:"callable",default:null,description:"right hand side function of ode-part of the system"},func_alg:{type:"callable",default:null,description:"output function of the system"},initial_value:{type:"any",default:null,description:"initial state / initial condition"},has_passthrough:{type:"boolean",default:"false",description:""},jac_dyn:{type:"any",default:null,description:"optional jacobian of `func_dyn` to improve convergence for implicit ode solvers"}},inputs:null,outputs:null},StateSpace:{blockClass:"StateSpace",description:"Linear time invariant (LTI) multi input multi output (MIMO) state space model.",docstringHtml:`<p>Linear time invariant (LTI) multi input multi output (MIMO) state space model.</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x} &amp;= \\mathbf{A} x + \\mathbf{B} u \\\\
           y &amp;= \\mathbf{C} x + \\mathbf{D} u
\\end{align}
\\end{equation*}
</div>
<p>where <cite>A</cite>, <cite>B</cite>, <cite>C</cite> and <cite>D</cite> are the state space matrices, <cite>x</cite> is the state,
<cite>u</cite> the input and <cite>y</cite> the output vector.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>A SISO state space block with two internal states can be initialized
like this:</p>
<pre class="code python literal-block">
<span class="name">S</span> <span class="operator">=</span> <span class="name">StateSpace</span><span class="punctuation">(</span><span class="whitespace">
</span>    <span class="name">A</span><span class="operator">=-</span><span class="name">np</span><span class="operator">.</span><span class="name">eye</span><span class="punctuation">(</span><span class="literal number integer">2</span><span class="punctuation">),</span><span class="whitespace">
</span>    <span class="name">B</span><span class="operator">=</span><span class="name">np</span><span class="operator">.</span><span class="name">ones</span><span class="punctuation">((</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">1</span><span class="punctuation">)),</span><span class="whitespace">
</span>    <span class="name">C</span><span class="operator">=</span><span class="name">np</span><span class="operator">.</span><span class="name">ones</span><span class="punctuation">((</span><span class="literal number integer">1</span><span class="punctuation">,</span> <span class="literal number integer">2</span><span class="punctuation">)),</span><span class="whitespace">
</span>    <span class="name">D</span><span class="operator">=</span><span class="literal number float">1.0</span><span class="whitespace">
</span>    <span class="punctuation">)</span>
</pre>
<p>and a MIMO (2 in, 2 out) state space block with three internal states
can be initialized like this:</p>
<pre class="code python literal-block">
<span class="name">S</span> <span class="operator">=</span> <span class="name">StateSpace</span><span class="punctuation">(</span><span class="whitespace">
</span>    <span class="name">A</span><span class="operator">=-</span><span class="name">np</span><span class="operator">.</span><span class="name">eye</span><span class="punctuation">(</span><span class="literal number integer">3</span><span class="punctuation">),</span><span class="whitespace">
</span>    <span class="name">B</span><span class="operator">=</span><span class="name">np</span><span class="operator">.</span><span class="name">ones</span><span class="punctuation">((</span><span class="literal number integer">3</span><span class="punctuation">,</span> <span class="literal number integer">2</span><span class="punctuation">)),</span><span class="whitespace">
</span>    <span class="name">C</span><span class="operator">=</span><span class="name">np</span><span class="operator">.</span><span class="name">ones</span><span class="punctuation">((</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">3</span><span class="punctuation">)),</span><span class="whitespace">
</span>    <span class="name">D</span><span class="operator">=</span><span class="name">np</span><span class="operator">.</span><span class="name">ones</span><span class="punctuation">((</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">2</span><span class="punctuation">))</span><span class="whitespace">
</span>    <span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>A, B, C, D <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>real valued state space matrices</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array_like, None</span></dt>
<dd>initial state / initial condition</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_dyn <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal dynamic operator for state equation</dd>
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">DynamicOperator</span></dt>
<dd>internal algebraic operator for mapping to outputs</dd>
</dl>
</div>
`,params:{A:{type:"any",default:null,description:""},B:{type:"any",default:null,description:""},C:{type:"any",default:null,description:""},D:{type:"any",default:null,description:"real valued state space matrices"},initial_value:{type:"any",default:null,description:"initial state / initial condition"}},inputs:null,outputs:null},PT1:{blockClass:"PT1",description:"First-order lag element (PT1).",docstringHtml:`<p>First-order lag element (PT1).</p>
<p>The transfer function is defined as</p>
<div class="math">
\\begin{equation*}
H(s) = \\frac{K}{1 + T s}
\\end{equation*}
</div>
<p>where <cite>K</cite> is the static gain and <cite>T</cite> is the time constant.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="name">pt1</span> <span class="operator">=</span> <span class="name">PT1</span><span class="punctuation">(</span><span class="name">K</span><span class="operator">=</span><span class="literal number float">2.0</span><span class="punctuation">,</span> <span class="name">T</span><span class="operator">=</span><span class="literal number float">0.5</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>K <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>static gain</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>time constant in seconds (must be &gt; 0)</dd>
</dl>
</div>
`,params:{K:{type:"any",default:null,description:"static gain"},T:{type:"any",default:null,description:"time constant in seconds (must be > 0)"}},inputs:["in"],outputs:["out"]},PT2:{blockClass:"PT2",description:"Second-order lag element (PT2).",docstringHtml:`<p>Second-order lag element (PT2).</p>
<p>The transfer function is defined as</p>
<div class="math">
\\begin{equation*}
H(s) = \\frac{K}{1 + 2 d T s + T^2 s^2}
\\end{equation*}
</div>
<p>where <cite>K</cite> is the static gain, <cite>T</cite> is the time constant
(related to the natural frequency by <span class="math">\\(\\omega_n = 1/T\\)</span>)
and <cite>d</cite> is the damping ratio.</p>
<p>The damping ratio <cite>d</cite> controls the transient behavior:</p>
<ul class="simple">
<li><span class="math">\\(d &lt; 1\\)</span>: underdamped (oscillatory)</li>
<li><span class="math">\\(d = 1\\)</span>: critically damped</li>
<li><span class="math">\\(d &gt; 1\\)</span>: overdamped</li>
</ul>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#underdamped second-order system</span><span class="whitespace">
</span><span class="name">pt2</span> <span class="operator">=</span> <span class="name">PT2</span><span class="punctuation">(</span><span class="name">K</span><span class="operator">=</span><span class="literal number float">1.0</span><span class="punctuation">,</span> <span class="name">T</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">,</span> <span class="name">d</span><span class="operator">=</span><span class="literal number float">0.3</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>K <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>static gain</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>time constant in seconds (must be &gt; 0)</dd>
<dt>d <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>damping ratio (must be &gt;= 0)</dd>
</dl>
</div>
`,params:{K:{type:"any",default:null,description:"static gain"},T:{type:"any",default:null,description:"time constant in seconds (must be > 0)"},d:{type:"any",default:null,description:"damping ratio (must be >= 0)"}},inputs:["in"],outputs:["out"]},LeadLag:{blockClass:"LeadLag",description:"Lead-Lag compensator.",docstringHtml:`<p>Lead-Lag compensator.</p>
<p>The transfer function is defined as</p>
<div class="math">
\\begin{equation*}
H(s) = K \\frac{T_1 s + 1}{T_2 s + 1}
\\end{equation*}
</div>
<p>where <cite>K</cite> is the static gain, <cite>T1</cite> is the lead time constant
and <cite>T2</cite> is the lag time constant.</p>
<ul class="simple">
<li><span class="math">\\(T_1 &gt; T_2\\)</span>: lead compensator (phase advance)</li>
<li><span class="math">\\(T_1 &lt; T_2\\)</span>: lag compensator (phase lag)</li>
<li><span class="math">\\(T_1 = T_2\\)</span>: pure gain</li>
</ul>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#lead compensator</span><span class="whitespace">
</span><span class="name">ll</span> <span class="operator">=</span> <span class="name">LeadLag</span><span class="punctuation">(</span><span class="name">K</span><span class="operator">=</span><span class="literal number float">1.0</span><span class="punctuation">,</span> <span class="name">T1</span><span class="operator">=</span><span class="literal number float">0.5</span><span class="punctuation">,</span> <span class="name">T2</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>K <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>static gain</dd>
<dt>T1 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>lead (numerator) time constant in seconds</dd>
<dt>T2 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>lag (denominator) time constant in seconds (must be &gt; 0)</dd>
</dl>
</div>
`,params:{K:{type:"any",default:null,description:"static gain"},T1:{type:"any",default:null,description:"lead (numerator) time constant in seconds"},T2:{type:"any",default:null,description:"lag (denominator) time constant in seconds (must be > 0)"}},inputs:["in"],outputs:["out"]},PID:{blockClass:"PID",description:"Proportional-Integral-Differentiation (PID) controller.",docstringHtml:`<p>Proportional-Integral-Differentiation (PID) controller.</p>
<p>The transfer function is defined as</p>
<div class="math">
\\begin{equation*}
H(s) = K_p + K_i \\frac{1}{s} + K_d \\frac{s}{1 + s / f_\\mathrm{max}}
\\end{equation*}
</div>
<p>where the differentiation is approximated by a high pass filter that holds
for signals up to a frequency of approximately <cite>f_max</cite>.</p>
<p>Internally realized as a linear state space model with two states
(differentiator filter state and integrator state).</p>
<div class="section" id="note">
<h3>Note</h3>
<p>Depending on <cite>f_max</cite>, the resulting system might become stiff or ill conditioned!
As a practical choice set <cite>f_max</cite> to 3x the highest expected signal frequency.
Since this block uses an approximation of real differentiation, the approximation will
not hold if there are high frequency components present in the signal. For example if
you have discontinuities such as steps or square waves.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#cutoff at 1kHz</span><span class="whitespace">
</span><span class="name">pid</span> <span class="operator">=</span> <span class="name">PID</span><span class="punctuation">(</span><span class="name">Kp</span><span class="operator">=</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="name">Ki</span><span class="operator">=</span><span class="literal number float">0.5</span><span class="punctuation">,</span> <span class="name">Kd</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">,</span> <span class="name">f_max</span><span class="operator">=</span><span class="literal number float">1e3</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Kp <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>proportional controller coefficient</dd>
<dt>Ki <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>integral controller coefficient</dd>
<dt>Kd <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>differentiator controller coefficient</dd>
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>highest expected signal frequency</dd>
</dl>
</div>
`,params:{Kp:{type:"any",default:null,description:"proportional controller coefficient"},Ki:{type:"any",default:null,description:"integral controller coefficient"},Kd:{type:"any",default:null,description:"differentiator controller coefficient"},f_max:{type:"any",default:null,description:"highest expected signal frequency"}},inputs:["in"],outputs:["out"]},AntiWindupPID:{blockClass:"AntiWindupPID",description:"Proportional-Integral-Differentiation (PID) controller with anti-windup mechanism (back-calculation).",docstringHtml:`<p>Proportional-Integral-Differentiation (PID) controller with anti-windup mechanism (back-calculation).</p>
<p>Anti-windup mechanisms are needed when the magnitude of the control signal
from the PID controller is limited by some real world saturation. In these cases,
the integrator will continue to accumulate the control error and &quot;wind itself up&quot;.
Once the setpoint is reached, this can result in significant overshoots. This
implementation adds a conditional feedback term to the internal integrator that
&quot;unwinds&quot; it when the PID output crosses some limits. This is pretty much a
deadzone feedback element for the integrator.</p>
<p>Mathematically, this block implements the following set of ODEs</p>
<div class="math">
\\begin{equation*}
\\begin{align}
\\dot{x}_1 &amp;= f_\\mathrm{max} (u - x_1) \\\\
\\dot{x}_2 &amp;= u - w
\\end{align}
\\end{equation*}
</div>
<p>with the anti-windup feedback (depending on the pid output)</p>
<div class="math">
\\begin{equation*}
w = K_s (y - \\min(\\max(y, y_\\mathrm{min}), y_\\mathrm{max}))
\\end{equation*}
</div>
<p>and the output itself</p>
<div class="math">
\\begin{equation*}
y = K_p u + K_d f_\\mathrm{max} (u - x_1) + K_i x_2
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>Depending on <cite>f_max</cite>, the resulting system might become stiff or ill conditioned!
As a practical choice set <cite>f_max</cite> to 3x the highest expected signal frequency.
Since this block uses an approximation of real differentiation, the approximation will
not hold if there are high frequency components present in the signal. For example if
you have discontinuities such as steps or square waves.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#cutoff at 1kHz, windup limits at [-5, 5]</span><span class="whitespace">
</span><span class="name">pid</span> <span class="operator">=</span> <span class="name">AntiWindupPID</span><span class="punctuation">(</span><span class="name">Kp</span><span class="operator">=</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="name">Ki</span><span class="operator">=</span><span class="literal number float">0.5</span><span class="punctuation">,</span> <span class="name">Kd</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">,</span> <span class="name">f_max</span><span class="operator">=</span><span class="literal number float">1e3</span><span class="punctuation">,</span> <span class="name">limits</span><span class="operator">=</span><span class="punctuation">[</span><span class="operator">-</span><span class="literal number integer">5</span><span class="punctuation">,</span> <span class="literal number integer">5</span><span class="punctuation">])</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Kp <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>proportional controller coefficient</dd>
<dt>Ki <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>integral controller coefficient</dd>
<dt>Kd <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>differentiator controller coefficient</dd>
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>highest expected signal frequency</dd>
<dt>Ks <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>feedback term for back calculation for anti-windup control of integrator</dd>
<dt>limits <span class="classifier-delimiter">:</span> <span class="classifier">array_like[float]</span></dt>
<dd>lower and upper limit for PID output that triggers anti-windup of integrator</dd>
</dl>
</div>
`,params:{Kp:{type:"number",default:"0.0",description:"proportional controller coefficient"},Ki:{type:"number",default:"0.0",description:"integral controller coefficient"},Kd:{type:"number",default:"0.0",description:"differentiator controller coefficient"},f_max:{type:"number",default:"100.0",description:"highest expected signal frequency"},Ks:{type:"number",default:"10.0",description:"feedback term for back calculation for anti-windup control of integrator"},limits:{type:"any",default:null,description:"lower and upper limit for PID output that triggers anti-windup of integrator"}},inputs:["in"],outputs:["out"]},RateLimiter:{blockClass:"RateLimiter",description:"Rate limiter block that limits the rate of change of a signal.",docstringHtml:`<p>Rate limiter block that limits the rate of change of a signal.</p>
<p>Implements a continuous-time rate limiter as a first-order tracking system
with clipped rate of change:</p>
<div class="math">
\\begin{equation*}
\\dot{x} = \\mathrm{clip}\\left(f_\\mathrm{max} (u - x),\\; -r,\\; r\\right)
\\end{equation*}
</div>
<p>where <cite>r</cite> is the maximum allowed rate and <cite>f_max</cite> controls the tracking
bandwidth when the signal is not rate-limited. The output is the state
<span class="math">\\(y = x\\)</span>.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>The parameter <cite>f_max</cite> should be set high enough that the output tracks
the input without lag when the rate is within limits.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#max rate of 10 units/s</span><span class="whitespace">
</span><span class="name">rl</span> <span class="operator">=</span> <span class="name">RateLimiter</span><span class="punctuation">(</span><span class="name">rate</span><span class="operator">=</span><span class="literal number float">10.0</span><span class="punctuation">,</span> <span class="name">f_max</span><span class="operator">=</span><span class="literal number float">1e3</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>rate <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>maximum rate of change (positive value)</dd>
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>tracking bandwidth parameter</dd>
</dl>
</div>
`,params:{rate:{type:"any",default:null,description:"maximum rate of change (positive value)"},f_max:{type:"any",default:null,description:"tracking bandwidth parameter"}},inputs:["in"],outputs:["out"]},Backlash:{blockClass:"Backlash",description:"Backlash (mechanical play) element.",docstringHtml:`<p>Backlash (mechanical play) element.</p>
<p>Models the hysteresis-like behavior of mechanical backlash in gears,
couplings and other systems with play. The output only tracks the input
after the input has moved through the full backlash width.</p>
<div class="math">
\\begin{equation*}
\\dot{x} = f_\\mathrm{max} \\left((u - x) - \\mathrm{clip}(u - x,\\; -w/2,\\; w/2)\\right)
\\end{equation*}
</div>
<p>where <cite>w</cite> is the total backlash width. Inside the dead zone <span class="math">\\(|u - x| \\leq w/2\\)</span>
the output does not move. Once the input pushes past the edge, the output
tracks with bandwidth <cite>f_max</cite>.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#backlash with 0.5 units of total play</span><span class="whitespace">
</span><span class="name">bl</span> <span class="operator">=</span> <span class="name">Backlash</span><span class="punctuation">(</span><span class="name">width</span><span class="operator">=</span><span class="literal number float">0.5</span><span class="punctuation">,</span> <span class="name">f_max</span><span class="operator">=</span><span class="literal number float">1e3</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>width <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>total backlash width (play)</dd>
<dt>f_max <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>tracking bandwidth parameter when engaged</dd>
</dl>
</div>
`,params:{width:{type:"any",default:null,description:"total backlash width (play)"},f_max:{type:"any",default:null,description:"tracking bandwidth parameter when engaged"}},inputs:["in"],outputs:["out"]},Deadband:{blockClass:"Deadband",description:"Deadband (dead zone) element.",docstringHtml:`<p>Deadband (dead zone) element.</p>
<p>Outputs zero when the input is within the dead zone, and passes
the signal shifted by the zone boundary otherwise:</p>
<div class="math">
\\begin{equation*}
y = \\begin{cases}
    u - u_\\mathrm{upper} &amp; \\text{if } u &gt; u_\\mathrm{upper} \\\\
    0 &amp; \\text{if } u_\\mathrm{lower} \\leq u \\leq u_\\mathrm{upper} \\\\
    u - u_\\mathrm{lower} &amp; \\text{if } u &lt; u_\\mathrm{lower}
\\end{cases}
\\end{equation*}
</div>
<p>or equivalently <span class="math">\\(y = u - \\mathrm{clip}(u,\\; u_\\mathrm{lower},\\; u_\\mathrm{upper})\\)</span>.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#symmetric dead zone of width 0.2</span><span class="whitespace">
</span><span class="name">db</span> <span class="operator">=</span> <span class="name">Deadband</span><span class="punctuation">(</span><span class="name">lower</span><span class="operator">=-</span><span class="literal number float">0.1</span><span class="punctuation">,</span> <span class="name">upper</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>lower <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>lower bound of the dead zone</dd>
<dt>upper <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>upper bound of the dead zone</dd>
</dl>
</div>
`,params:{lower:{type:"any",default:null,description:"lower bound of the dead zone"},upper:{type:"any",default:null,description:"upper bound of the dead zone"}},inputs:["in"],outputs:["out"]},TransferFunctionNumDen:{blockClass:"TransferFunctionNumDen",description:"This block defines a LTI (SISO) transfer function.",docstringHtml:`<p>This block defines a LTI (SISO) transfer function.</p>
<p>The transfer function is defined in polynomial (numerator-denominator) form</p>
<div class="math">
\\begin{equation*}
\\mathbf{H}(s) = \\frac{b_n + b_{n-1} s + \\dots + b_{0} s^n}{a_m + a_{m-1} s + \\dots + a_{0} s^m}
\\end{equation*}
</div>
<p>where <cite>Num</cite> is the list of numerator polynomial coefficients and <cite>Den</cite> the
list of denominator coefficients.</p>
<p>Upon initialization, the state space realization of the transfer function is
computed using <cite>scipy.signal.TransferFunction(Num, Den).to_ss()</cite>.</p>
<p>The resulting state space model of the form</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x} &amp;= \\mathbf{A} x + \\mathbf{B} u \\\\
           y &amp;= \\mathbf{C} x + \\mathbf{D} u
\\end{align}
\\end{equation*}
</div>
<p>is handled the same as the 'StateSpace' block, where <cite>A</cite>, <cite>B</cite>, <cite>C</cite> and <cite>D</cite>
are the state space matrices, <cite>x</cite> is the internal state, <cite>u</cite> the input and
<cite>y</cite> the output vector.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Num <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>numerator polynomial coefficients</dd>
<dt>Den <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>denominator polynomial coefficients</dd>
</dl>
</div>
`,params:{Num:{type:"any",default:null,description:"numerator polynomial coefficients"},Den:{type:"any",default:null,description:"denominator polynomial coefficients"}},inputs:["in"],outputs:["out"]},TransferFunctionZPG:{blockClass:"TransferFunctionZPG",description:"This block defines a LTI (SISO) transfer function.",docstringHtml:`<p>This block defines a LTI (SISO) transfer function.</p>
<p>The transfer function is defined in zeros-poles-gain (ZPG) form</p>
<div class="math">
\\begin{equation*}
\\mathbf{H}(s) = k \\frac{(s - z_1)(s - z_2)\\cdots(s - z_m)}{(s - p_1)(s - p_2)\\cdots(s - p_n)}
\\end{equation*}
</div>
<p>where <cite>Zeros</cite> are the scalar (possibly complex conjugate) zeros of the
transfer function, and <cite>Poles</cite> are the poles (denominator zeros) of the
transfer function. <cite>Gain</cite> is the scalar factor <cite>k</cite>.</p>
<p>Upon initialization, the state space realization of the transfer function is
computed using <cite>scipy.signal.ZerosPolesGain(Zeros, Poles, Gain).to_ss()</cite>.</p>
<p>The resulting state space model of the form</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    \\dot{x} &amp;= \\mathbf{A} x + \\mathbf{B} u \\\\
           y &amp;= \\mathbf{C} x + \\mathbf{D} u
\\end{align}
\\end{equation*}
</div>
<p>is handled the same as the 'StateSpace' block, where <cite>A</cite>, <cite>B</cite>, <cite>C</cite> and <cite>D</cite>
are the state space matrices, <cite>x</cite> is the internal state, <cite>u</cite> the input and
<cite>y</cite> the output vector.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Poles <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>transfer function poles</dd>
<dt>Zeros <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>transfer function zeros</dd>
<dt>Gain <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>gain term of transfer function</dd>
</dl>
</div>
`,params:{Zeros:{type:"any",default:null,description:"transfer function zeros"},Poles:{type:"any",default:null,description:"transfer function poles"},Gain:{type:"number",default:"1.0",description:"gain term of transfer function"}},inputs:["in"],outputs:["out"]},ButterworthLowpassFilter:{blockClass:"ButterworthLowpassFilter",description:"Direct implementation of a low pass butterworth filter block.",docstringHtml:`<p>Direct implementation of a low pass butterworth filter block.</p>
<p>Follows the same structure as the 'StateSpace' block in the
'pathsim.blocks' module. The numerator and denominator of the
filter transfer function are generated and then the transfer
function is realized as a state space model.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Fc <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>corner frequency of the filter in [Hz]</dd>
<dt>n <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>filter order</dd>
</dl>
</div>
`,params:{Fc:{type:"number",default:"100.0",description:"corner frequency of the filter in [Hz]"},n:{type:"integer",default:"2",description:"filter order"}},inputs:["in"],outputs:["out"]},ButterworthHighpassFilter:{blockClass:"ButterworthHighpassFilter",description:"Direct implementation of a high pass butterworth filter block.",docstringHtml:`<p>Direct implementation of a high pass butterworth filter block.</p>
<p>Follows the same structure as the 'StateSpace' block in the
'pathsim.blocks' module. The numerator and denominator of the
filter transfer function are generated and then the transfer
function is realized as a state space model.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Fc <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>corner frequency of the filter in [Hz]</dd>
<dt>n <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>filter order</dd>
</dl>
</div>
`,params:{Fc:{type:"number",default:"100.0",description:"corner frequency of the filter in [Hz]"},n:{type:"integer",default:"2",description:"filter order"}},inputs:["in"],outputs:["out"]},ButterworthBandpassFilter:{blockClass:"ButterworthBandpassFilter",description:"Direct implementation of a bandpass butterworth filter block.",docstringHtml:`<p>Direct implementation of a bandpass butterworth filter block.</p>
<p>Follows the same structure as the 'StateSpace' block in the
'pathsim.blocks' module. The numerator and denominator of the
filter transfer function are generated and then the transfer
function is realized as a state space model.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Fc <span class="classifier-delimiter">:</span> <span class="classifier">list[float]</span></dt>
<dd>corner frequencies (left, right) of the filter in [Hz]</dd>
<dt>n <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>filter order</dd>
</dl>
</div>
`,params:{Fc:{type:"any",default:null,description:"corner frequencies (left, right) of the filter in [Hz]"},n:{type:"integer",default:"2",description:"filter order"}},inputs:["in"],outputs:["out"]},ButterworthBandstopFilter:{blockClass:"ButterworthBandstopFilter",description:"Direct implementation of a bandstop butterworth filter block.",docstringHtml:`<p>Direct implementation of a bandstop butterworth filter block.</p>
<p>Follows the same structure as the 'StateSpace' block in the
'pathsim.blocks' module. The numerator and denominator of the
filter transfer function are generated and then the transfer
function is realized as a state space model.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Fc <span class="classifier-delimiter">:</span> <span class="classifier">tuple[float], list[float]</span></dt>
<dd>corner frequencies (left, right) of the filter in [Hz]</dd>
<dt>n <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>filter order</dd>
</dl>
</div>
`,params:{Fc:{type:"any",default:null,description:"corner frequencies (left, right) of the filter in [Hz]"},n:{type:"integer",default:"2",description:"filter order"}},inputs:["in"],outputs:["out"]},MassMatrixDAE:{blockClass:"MassMatrixDAE",description:"Mass-matrix DAE block.",docstringHtml:`<p>Mass-matrix DAE block.</p>
<p>Solves an implicit ODE with a (possibly singular) constant mass matrix:</p>
<div class="math">
\\begin{equation*}
\\mathbf{M} \\, \\dot{x} = f(x, u, t), \\quad y = x
\\end{equation*}
</div>
<p><cite>f</cite> has the same signature as in :class:\`ODE\` - only the way the solver
integrates it differs. The mass matrix <cite>M</cite> is stored on the block and
installed into the solver's stage builder when an implicit solver is
attached. Explicit solvers see a pure ODE and will silently produce wrong
results for singular <cite>M</cite> - use one of the ESDIRK/DIRK/EUB families for any
non-trivial mass.</p>
<p>The JIT path traces <cite>func</cite> and derives <span class="math">\\(\\partial f / \\partial x\\)</span>
analytically via auto-differentiation when no analytical Jacobian is
supplied.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>right-hand side <tt class="docutils literal">f(x, u, t) <span class="pre">-&gt;</span> ndarray</tt></dd>
<dt>mass <span class="classifier-delimiter">:</span> <span class="classifier">Mass</span></dt>
<dd>mass matrix descriptor (dense, banded or sparse)</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>initial state, must have the same length as <tt class="docutils literal">mass.n</tt></dd>
<dt>jac <span class="classifier-delimiter">:</span> <span class="classifier">callable, optional</span></dt>
<dd>analytical <span class="math">\\(\\partial f / \\partial x\\)</span> as a flat row-major
<tt class="docutils literal">n × n</tt> array. If omitted, numerical or AD-derived Jacobians are
used downstream.</dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"right-hand side ``f(x, u, t) -> ndarray``"},mass:{type:"any",default:null,description:"mass matrix descriptor (dense, banded or sparse)"},initial_value:{type:"any",default:null,description:"initial state, must have the same length as ``mass.n``"},jac:{type:"any",default:null,description:"analytical :math:`\\partial f / \\partial x` as a flat row-major ``n × n`` array. If omitted, numerical or AD-derived Jacobians are used downstream."}},inputs:null,outputs:null},SemiExplicitDAE:{blockClass:"SemiExplicitDAE",description:"Semi-explicit Index-1 DAE block.",docstringHtml:`<p>Semi-explicit Index-1 DAE block.</p>
<p>Solves an Index-1 system with split differential and algebraic states</p>
<div class="math">
\\begin{align*}
\\dot{x} &amp;= f_\\mathrm{dyn}(x, z, u, t) \\\\
0       &amp;= f_\\mathrm{alg}(x, z, u, t)
\\end{align*}
</div>
<p>The algebraic state <span class="math">\\(z\\)</span> is eliminated by an inner Newton on
<span class="math">\\(f_\\mathrm{alg}(x, z, u, t) = 0\\)</span> at every RHS evaluation
(warmstarted from the previous call). The outer solver sees a plain ODE
in <span class="math">\\(x\\)</span>, so any of the explicit or implicit solvers in fastsim can
be attached.</p>
<p>The block output is <span class="math">\\([x; z]\\)</span> (with <cite>z</cite> taken from the converged
inner Newton), so downstream blocks see both differential and algebraic
states.</p>
<p>Trade-offs vs formulating the same system as a :class:\`MassMatrixDAE\`
with a block-diagonal singular mass:</p>
<ul class="simple">
<li>explicit solvers (RKDP54, RKF78, RKV65, …) work</li>
<li>smaller Newton problem per stage (size <span class="math">\\(n_z\\)</span> instead of
<span class="math">\\(n_x + n_z\\)</span>)</li>
<li>inner Newton cost per RHS call (typically 1–3 iterations once
warmstarted)</li>
<li>adaptive error control watches only <span class="math">\\(x\\)</span>, not <span class="math">\\(z\\)</span></li>
</ul>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>f_dyn <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>differential RHS <tt class="docutils literal">f_dyn(x, z, u, t) <span class="pre">-&gt;</span> ndarray</tt> of length <tt class="docutils literal">n_x</tt></dd>
<dt>f_alg <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>algebraic constraint <tt class="docutils literal">f_alg(x, z, u, t) <span class="pre">-&gt;</span> ndarray</tt> of length <tt class="docutils literal">n_z</tt></dd>
<dt>x0 <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>initial differential state (length <tt class="docutils literal">n_x</tt>)</dd>
<dt>z0 <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>initial algebraic state (length <tt class="docutils literal">n_z</tt>), used as Newton warmstart</dd>
<dt>jac_z <span class="classifier-delimiter">:</span> <span class="classifier">callable, optional</span></dt>
<dd>analytical <span class="math">\\(\\partial f_\\mathrm{alg} / \\partial z\\)</span> as a flat
row-major <tt class="docutils literal">n_z × n_z</tt> array. Falls back to central differences if
omitted.</dd>
</dl>
</div>
`,params:{f_dyn:{type:"any",default:null,description:"differential RHS ``f_dyn(x, z, u, t) -> ndarray`` of length ``n_x``"},f_alg:{type:"any",default:null,description:"algebraic constraint ``f_alg(x, z, u, t) -> ndarray`` of length ``n_z``"},x0:{type:"any",default:null,description:"initial differential state (length ``n_x``)"},z0:{type:"any",default:null,description:"initial algebraic state (length ``n_z``), used as Newton warmstart"},jac_z:{type:"any",default:null,description:"analytical :math:`\\partial f_\\mathrm{alg} / \\partial z` as a flat row-major ``n_z × n_z`` array. Falls back to central differences if omitted."}},inputs:null,outputs:null},FullyImplicitDAE:{blockClass:"FullyImplicitDAE",description:"Fully-implicit DAE block.",docstringHtml:`<p>Fully-implicit DAE block.</p>
<p>For systems that can't be cast into semi-explicit or mass-matrix form -
implicit constitutive relations, mixed differential/algebraic with
non-trivial coupling - the residual form</p>
<div class="math">
\\begin{equation*}
F(x, \\dot{x}, u, t) = 0, \\quad y = x
\\end{equation*}
</div>
<p>is solved directly. Only implicit solvers (ESDIRK/DIRK family) work; the
block installs a fully-implicit stage builder into the engine via the
post-processing hook.</p>
<p>The JIT path traces <cite>func</cite> and derives both
<span class="math">\\(\\partial F / \\partial x\\)</span> and <span class="math">\\(\\partial F / \\partial \\dot{x}\\)</span>
via auto-differentiation when no analytical Jacobians are supplied.
Index-1 systems (singular <span class="math">\\(\\partial F / \\partial \\dot{x}\\)</span>):
prefer DIRK over ESDIRK for stability.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>residual <tt class="docutils literal">F(x, xdot, u, t) <span class="pre">-&gt;</span> ndarray</tt></dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>consistent <span class="math">\\(x_0\\)</span>. The caller is responsible for choosing it such
that there exists an <span class="math">\\(\\dot{x}_0\\)</span> with
<span class="math">\\(F(x_0, \\dot{x}_0, u_0, 0) \\approx 0\\)</span>.</dd>
<dt>jac_x <span class="classifier-delimiter">:</span> <span class="classifier">callable, optional</span></dt>
<dd>analytical <span class="math">\\(\\partial F / \\partial x\\)</span> as a flat row-major
<tt class="docutils literal">n × n</tt> array. Falls back to numerical (central differences) if
omitted.</dd>
<dt>jac_xdot <span class="classifier-delimiter">:</span> <span class="classifier">callable, optional</span></dt>
<dd>analytical <span class="math">\\(\\partial F / \\partial \\dot{x}\\)</span> as a flat row-major
<tt class="docutils literal">n × n</tt> array. Falls back to numerical (central differences) if
omitted.</dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"residual ``F(x, xdot, u, t) -> ndarray``"},initial_value:{type:"any",default:null,description:"consistent :math:`x_0`. The caller is responsible for choosing it such that there exists an :math:`\\dot{x}_0` with :math:`F(x_0, \\dot{x}_0, u_0, 0) \\approx 0`."},jac_x:{type:"any",default:null,description:"analytical :math:`\\partial F / \\partial x` as a flat row-major ``n × n`` array. Falls back to numerical (central differences) if omitted."},jac_xdot:{type:"any",default:null,description:"analytical :math:`\\partial F / \\partial \\dot{x}` as a flat row-major ``n × n`` array. Falls back to numerical (central differences) if omitted."}},inputs:null,outputs:null},Adder:{blockClass:"Adder",description:"Summs / adds up all input signals to a single output signal (MISO)",docstringHtml:`<p>Summs / adds up all input signals to a single output signal (MISO)</p>
<p>This is how it works in the default case</p>
<div class="math">
\\begin{equation*}
y(t) = \\sum_i u_i(t)
\\end{equation*}
</div>
<p>and like this when additional operations are defined</p>
<div class="math">
\\begin{equation*}
y(t) = \\sum_i \\mathrm{op}_i \\cdot u_i(t)
\\end{equation*}
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>This is the default initialization that just adds up all the inputs:</p>
<pre class="code python literal-block">
<span class="name">A</span> <span class="operator">=</span> <span class="name">Adder</span><span class="punctuation">()</span>
</pre>
<p>and this is the initialization with specific operations that subtracts
the second from first input and neglects all others:</p>
<pre class="code python literal-block">
<span class="name">A</span> <span class="operator">=</span> <span class="name">Adder</span><span class="punctuation">(</span><span class="literal string single">'+-'</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its operation (<cite>op_alg</cite>) will be called
multiple times per timestep, each time when <cite>Simulation._update(t)</cite> is
called in the global simulation loop.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>operations <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span></dt>
<dd>optional string of operations to be applied before
summation, i.e. '+-' will compute the difference,
'None' will just perform regular sum</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>_ops <span class="classifier-delimiter">:</span> <span class="classifier">dict</span></dt>
<dd>dict that maps string operations to numerical</dd>
<dt>_ops_array <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>operations converted to array</dd>
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{operations:{type:"any",default:null,description:"optional string of operations to be applied before summation, i.e. '+-' will compute the difference, 'None' will just perform regular sum"}},inputs:null,outputs:["out"]},Multiplier:{blockClass:"Multiplier",description:"Multiplies all signals from all input ports (MISO).",docstringHtml:`<p>Multiplies all signals from all input ports (MISO).</p>
<div class="math">
\\begin{equation*}
y(t) = \\prod_i u_i(t)
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its operation (<cite>op_alg</cite>) will be called
multiple times per timestep, each time when <cite>Simulation._update(t)</cite> is
called in the global simulation loop.</p>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator that wraps 'prod'</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:["out"]},Divider:{blockClass:"Divider",description:"Multiplies and divides input signals (MISO).",docstringHtml:`<p>Multiplies and divides input signals (MISO).</p>
<p>This is the default behavior (multiply all):</p>
<div class="math">
\\begin{equation*}
y(t) = \\prod_i u_i(t)
\\end{equation*}
</div>
<p>and this is the behavior with an operations string:</p>
<div class="math">
\\begin{equation*}
y(t) = \\frac{\\prod_{i \\in M} u_i(t)}{\\prod_{j \\in D} u_j(t)}
\\end{equation*}
</div>
<p>where <span class="math">\\(M\\)</span> is the set of inputs with <tt class="docutils literal">*</tt> and <span class="math">\\(D\\)</span> the set with <tt class="docutils literal">/</tt>.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>Default initialization multiplies the first input and divides by the second:</p>
<pre class="code python literal-block">
<span class="name">D</span> <span class="operator">=</span> <span class="name">Divider</span><span class="punctuation">()</span>
</pre>
<p>Multiply the first two inputs and divide by the third:</p>
<pre class="code python literal-block">
<span class="name">D</span> <span class="operator">=</span> <span class="name">Divider</span><span class="punctuation">(</span><span class="literal string single">'**/'</span><span class="punctuation">)</span>
</pre>
<p>Raise an error instead of producing <tt class="docutils literal">inf</tt> when a denominator input is zero:</p>
<pre class="code python literal-block">
<span class="name">D</span> <span class="operator">=</span> <span class="name">Divider</span><span class="punctuation">(</span><span class="literal string single">'**/'</span><span class="punctuation">,</span> <span class="name">zero_div</span><span class="operator">=</span><span class="literal string single">'raise'</span><span class="punctuation">)</span>
</pre>
<p>Clamp the denominator to machine epsilon so the output stays finite:</p>
<pre class="code python literal-block">
<span class="name">D</span> <span class="operator">=</span> <span class="name">Divider</span><span class="punctuation">(</span><span class="literal string single">'**/'</span><span class="punctuation">,</span> <span class="name">zero_div</span><span class="operator">=</span><span class="literal string single">'clamp'</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its operation (<tt class="docutils literal">op_alg</tt>) will be called
multiple times per timestep, each time when <tt class="docutils literal">Simulation._update(t)</tt> is
called in the global simulation loop.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>operations <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span></dt>
<dd>String of <tt class="docutils literal">*</tt> and <tt class="docutils literal">/</tt> characters indicating which inputs are
multiplied (<tt class="docutils literal">*</tt>) or divided (<tt class="docutils literal">/</tt>). Inputs beyond the length of
the string default to <tt class="docutils literal">*</tt>. Defaults to <tt class="docutils literal"><span class="pre">'*/'</span></tt> (divide second
input by first).</dd>
<dt>zero_div <span class="classifier-delimiter">:</span> <span class="classifier">str, optional</span></dt>
<dd><p class="first">Behaviour when a denominator input is zero. One of:</p>
<dl class="last docutils">
<dt><tt class="docutils literal">'warn'</tt> <em>(default)</em></dt>
<dd>Propagates <tt class="docutils literal">inf</tt> and emits a <tt class="docutils literal">RuntimeWarning</tt> - numpy's
standard behaviour.</dd>
<dt><tt class="docutils literal">'raise'</tt></dt>
<dd>Raises <tt class="docutils literal">ZeroDivisionError</tt>.</dd>
<dt><tt class="docutils literal">'clamp'</tt></dt>
<dd>Clamps the denominator magnitude to machine epsilon
(<tt class="docutils literal"><span class="pre">numpy.finfo(float).eps</span></tt>), preserving sign, so the output
stays large-but-finite rather than <tt class="docutils literal">inf</tt>.</dd>
</dl>
</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>_ops <span class="classifier-delimiter">:</span> <span class="classifier">dict</span></dt>
<dd>Maps operation characters to exponent values (<tt class="docutils literal">+1</tt> or <tt class="docutils literal"><span class="pre">-1</span></tt>).</dd>
<dt>_ops_array <span class="classifier-delimiter">:</span> <span class="classifier">numpy.ndarray</span></dt>
<dd>Exponents (+1 for <tt class="docutils literal">*</tt>, -1 for <tt class="docutils literal">/</tt>) converted to an array.</dd>
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>Internal algebraic operator.</dd>
</dl>
</div>
`,params:{operations:{type:"string",default:'"*/"',description:"String of ``*`` and ``/`` characters indicating which inputs are multiplied (``*``) or divided (``/``). Inputs beyond the length of the string default to ``*``. Defaults to ``'*/'`` (divide second input by first)."},zero_div:{type:"string",default:'"warn"',description:"Behaviour when a denominator input is zero. One of:"}},inputs:null,outputs:["out"]},Amplifier:{blockClass:"Amplifier",description:"Amplifies the input signal by multiplication with a constant gain term.",docstringHtml:`<p>Amplifies the input signal by multiplication with a constant gain term.</p>
<p>Like this:</p>
<div class="math">
\\begin{equation*}
y(t) = \\mathrm{gain} \\cdot u(t)
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its operation (<cite>op_alg</cite>) will be called
multiple times per timestep, each time when <cite>Simulation._update(t)</cite> is
called in the global simulation loop.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#amplification by factor 5</span><span class="whitespace">
</span><span class="name">A</span> <span class="operator">=</span> <span class="name">Amplifier</span><span class="punctuation">(</span><span class="name">gain</span><span class="operator">=</span><span class="literal number integer">5</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>gain <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>amplifier gain</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{gain:{type:"any",default:null,description:"amplifier gain"}},inputs:null,outputs:null},Function:{blockClass:"Function",description:"Arbitrary MIMO function block, defined by a function or `lambda` expression.",docstringHtml:`<p>Arbitrary MIMO function block, defined by a function or <cite>lambda</cite> expression.</p>
<p>The function can have multiple arguments that are then provided
by the input channels of the function block.</p>
<p>Form multi input, the function has to specify multiple arguments
and for multi output, the aoutputs have to be provided as a
tuple or list.</p>
<p>In the context of the global system, this block implements algebraic
components of the global system ODE/DAE.</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\mathrm{func}(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block is purely algebraic and its operation (<cite>op_alg</cite>) will be called
multiple times per timestep, each time when <cite>Simulation._update(t)</cite> is
called in the global simulation loop.
Therefore <cite>func</cite> must be purely algebraic and not introduce states,
delay, etc. For interfacing with external stateful APIs, use the
<cite>Wrapper</cite> block.</p>
</div>
<div class="section" id="note-1">
<h3>Note</h3>
<p>If the outputs are provided as a single numpy array, they are
considered a single output. For MIMO, output has to be tuple.</p>
</div>
<div class="section" id="example">
<h3>Example</h3>
<p>consider the function:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Function</span><span class="whitespace">

</span><span class="keyword">def</span> <span class="name function">f</span><span class="punctuation">(</span><span class="name">a</span><span class="punctuation">,</span> <span class="name">b</span><span class="punctuation">,</span> <span class="name">c</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">a</span><span class="operator">**</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="name">a</span><span class="operator">*</span><span class="name">b</span><span class="punctuation">,</span> <span class="name">b</span><span class="operator">/</span><span class="name">c</span><span class="whitespace">

</span><span class="name">fn</span> <span class="operator">=</span> <span class="name">Function</span><span class="punctuation">(</span><span class="name">f</span><span class="punctuation">)</span>
</pre>
<p>then, when the block is updated, the input channels of the block are
assigned to the function arguments following this scheme:</p>
<pre class="code literal-block">
inputs[0] -&gt; a
inputs[1] -&gt; b
inputs[2] -&gt; c
</pre>
<p>and the function outputs are assigned to the
output channels of the block in the same way:</p>
<pre class="code literal-block">
a**2 -&gt; outputs[0]
a*b  -&gt; outputs[1]
b/c  -&gt; outputs[2]
</pre>
<p>Because the <cite>Function</cite> block only has a single argument, it can be
used to decorate a function and make it a <cite>PathSim</cite> block. This might
be handy in some cases to keep definitions concise and localized
in the code:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Function</span><span class="whitespace">

</span><span class="comment single">#does the same as the definition above</span><span class="whitespace">

</span><span class="name decorator">&#64;Function</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">fn</span><span class="punctuation">(</span><span class="name">a</span><span class="punctuation">,</span> <span class="name">b</span><span class="punctuation">,</span> <span class="name">c</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">a</span><span class="operator">**</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="name">a</span><span class="operator">*</span><span class="name">b</span><span class="punctuation">,</span> <span class="name">b</span><span class="operator">/</span><span class="name">c</span><span class="whitespace">

</span><span class="comment single">#'fn' is now a PathSim block</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>MIMO function that defines algebraic block IO behaviour, signature <cite>func(*tuple)</cite></dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator that wraps <cite>func</cite></dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"MIMO function that defines algebraic block IO behaviour, signature `func(*tuple)`"}},inputs:null,outputs:null},Polynomial:{blockClass:"Polynomial",description:"Polynomial operator block.",docstringHtml:`<p>Polynomial operator block.</p>
<p>Evaluates a polynomial in the input. The coefficients follow the
<cite>numpy.polyval</cite> convention, with the highest order term first:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = c_0 \\vec{u}^n + c_1 \\vec{u}^{n-1} + \\dots + c_{n-1} \\vec{u} + c_n
\\end{equation*}
</div>
<p>This block supports vector inputs (the polynomial is evaluated
element-wise).</p>
<div class="section" id="example">
<h3>Example</h3>
<p>Quadratic <span class="math">\\(y = 2 u^2 + 3 u + 1\\)</span>:</p>
<pre class="code python literal-block">
<span class="name">p</span> <span class="operator">=</span> <span class="name">Polynomial</span><span class="punctuation">(</span><span class="name">coeffs</span><span class="operator">=</span><span class="punctuation">[</span><span class="literal number integer">2</span><span class="punctuation">,</span> <span class="literal number integer">3</span><span class="punctuation">,</span> <span class="literal number integer">1</span><span class="punctuation">])</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>coeffs <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>polynomial coefficients in descending order of power,
following the <tt class="docutils literal">numpy.polyval</tt> convention</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{coeffs:{type:"any",default:null,description:"polynomial coefficients in descending order of power, following the ``numpy.polyval`` convention"}},inputs:null,outputs:null},Sin:{blockClass:"Sin",description:"Sine operator block.",docstringHtml:`<p>Sine operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\sin(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Cos:{blockClass:"Cos",description:"Cosine operator block.",docstringHtml:`<p>Cosine operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\cos(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Tan:{blockClass:"Tan",description:"Tangent operator block.",docstringHtml:`<p>Tangent operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\tan(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Tanh:{blockClass:"Tanh",description:"Hyperbolic tangent operator block.",docstringHtml:`<p>Hyperbolic tangent operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\tanh(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Abs:{blockClass:"Abs",description:"Absolute value operator block.",docstringHtml:`<p>Absolute value operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\vert| \\vec{u} \\vert|
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Sqrt:{blockClass:"Sqrt",description:"Square root operator block.",docstringHtml:`<p>Square root operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\sqrt{|\\vec{u}|}
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Exp:{blockClass:"Exp",description:"Exponential operator block.",docstringHtml:`<p>Exponential operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = e^{\\vec{u}}
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Log:{blockClass:"Log",description:"Natural logarithm operator block.",docstringHtml:`<p>Natural logarithm operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\ln(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Log10:{blockClass:"Log10",description:"Base-10 logarithm operator block.",docstringHtml:`<p>Base-10 logarithm operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\log_{10}(\\vec{u})
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Mod:{blockClass:"Mod",description:"Modulo operator block.",docstringHtml:`<p>Modulo operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\vec{u} \\bmod m
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>modulo is not differentiable at discontinuities</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>modulus <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>modulus value</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{modulus:{type:"any",default:null,description:"modulus value Attributes ----------"}},inputs:null,outputs:null},Clip:{blockClass:"Clip",description:"Clipping/saturation operator block.",docstringHtml:`<p>Clipping/saturation operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\text{clip}(\\vec{u}, u_{min}, u_{max})
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>min_val <span class="classifier-delimiter">:</span> <span class="classifier">float, array_like</span></dt>
<dd>minimum clipping value</dd>
<dt>max_val <span class="classifier-delimiter">:</span> <span class="classifier">float, array_like</span></dt>
<dd>maximum clipping value</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{min_val:{type:"any",default:null,description:"minimum clipping value"},max_val:{type:"any",default:null,description:"maximum clipping value Attributes ----------"}},inputs:null,outputs:null},Pow:{blockClass:"Pow",description:"Raise to power operator block.",docstringHtml:`<p>Raise to power operator block.</p>
<p>This block supports vector inputs. This is the operation it does:</p>
<div class="math">
\\begin{equation*}
\\vec{y} = \\vec{u}^{p}
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>exponent <span class="classifier-delimiter">:</span> <span class="classifier">float, array_like</span></dt>
<dd>exponent to raise the input to the power of</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{exponent:{type:"any",default:null,description:"exponent to raise the input to the power of Attributes ----------"}},inputs:null,outputs:null},Atan2:{blockClass:"Atan2",description:"Two-argument arctangent block.",docstringHtml:`<p>Two-argument arctangent block.</p>
<p>Computes the four-quadrant arctangent of two inputs:</p>
<div class="math">
\\begin{equation*}
y = \\mathrm{atan2}(a, b)
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>This block takes exactly two inputs (a, b) and produces one output.
The first input is the y-coordinate, the second is the x-coordinate,
matching the convention of <tt class="docutils literal">numpy.arctan2(y, x)</tt>.</p>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:["a","b"],outputs:["y"]},Rescale:{blockClass:"Rescale",description:"Linear rescaling / mapping block.",docstringHtml:`<p>Linear rescaling / mapping block.</p>
<p>Maps the input linearly from range <tt class="docutils literal">[i0, i1]</tt> to range <tt class="docutils literal">[o0, o1]</tt>.
Optionally saturates the output to <tt class="docutils literal">[o0, o1]</tt>.</p>
<div class="math">
\\begin{equation*}
y = o_0 + \\frac{(x - i_0) \\cdot (o_1 - o_0)}{i_1 - i_0}
\\end{equation*}
</div>
<p>This block supports vector inputs.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>i0 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>input range lower bound</dd>
<dt>i1 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>input range upper bound</dd>
<dt>o0 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>output range lower bound</dd>
<dt>o1 <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>output range upper bound</dd>
<dt>saturate <span class="classifier-delimiter">:</span> <span class="classifier">bool</span></dt>
<dd>if True, clamp output to [min(o0,o1), max(o0,o1)]</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{i0:{type:"number",default:"0.0",description:"input range lower bound"},i1:{type:"number",default:"1.0",description:"input range upper bound"},o0:{type:"number",default:"0.0",description:"output range lower bound"},o1:{type:"number",default:"1.0",description:"output range upper bound"},saturate:{type:"boolean",default:"false",description:"if True, clamp output to [min(o0,o1), max(o0,o1)]"}},inputs:null,outputs:null},Alias:{blockClass:"Alias",description:"Signal alias / pass-through block.",docstringHtml:`<p>Signal alias / pass-through block.</p>
<p>Passes the input directly to the output without modification.
This is useful for signal renaming in model composition.</p>
<div class="math">
\\begin{equation*}
y = x
\\end{equation*}
</div>
<p>This block supports vector inputs.</p>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},Switch:{blockClass:"Switch",description:"Switch block that selects between its inputs.",docstringHtml:`<p>Switch block that selects between its inputs.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>The block is initialized like this:</p>
<pre class="code python literal-block">
<span class="comment single">#default None -&gt; no passthrough</span><span class="whitespace">
</span><span class="name">s1</span> <span class="operator">=</span> <span class="name">Switch</span><span class="punctuation">()</span><span class="whitespace">

</span><span class="comment single">#selecting port 2 as passthrough</span><span class="whitespace">
</span><span class="name">s2</span> <span class="operator">=</span> <span class="name">Switch</span><span class="punctuation">(</span><span class="literal number integer">2</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="comment single">#change the state of the switch to port 3</span><span class="whitespace">
</span><span class="name">s2</span><span class="operator">.</span><span class="name">select</span><span class="punctuation">(</span><span class="literal number integer">3</span><span class="punctuation">)</span>
</pre>
<p>Sets block output depending on <cite>self.switch_state</cite> like this:</p>
<pre class="code literal-block">
switch_state == None -&gt; outputs[0] = 0

switch_state == 0 -&gt; outputs[0] = inputs[0]

switch_state == 1 -&gt; outputs[0] = inputs[1]

switch_state == 2 -&gt; outputs[0] = inputs[2]

...
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>switch_state <span class="classifier-delimiter">:</span> <span class="classifier">int, None</span></dt>
<dd>state of the switch</dd>
</dl>
</div>
`,params:{switch_state:{type:"any",default:null,description:"state of the switch"}},inputs:null,outputs:["out"]},LUT1D:{blockClass:"LUT1D",description:"One-dimensional lookup table with linear interpolation functionality.",docstringHtml:`<p>One-dimensional lookup table with linear interpolation functionality.</p>
<p>This class implements a 1-dimensional lookup table that uses scipy's interp1d <a class="footnote-reference" href="#scipy" id="footnote-reference-1">[1]</a>
for piecewise linear interpolation along a single axis. The interpolation
provides linear interpolation between adjacent data points and supports
extrapolation beyond the input data range using the 'extrapolate' fill mode.</p>
<p>The LUT1D acts as a Function block.</p>
<div class="section" id="references">
<h3>References</h3>
<table class="docutils footnote" frame="void" id="scipy" rules="none">
<colgroup><col class="label" /><col /></colgroup>
<tbody valign="top">
<tr><td class="label"><a class="fn-backref" href="#footnote-reference-1">[1]</a></td><td><a class="reference external" href="https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html">https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html</a></td></tr>
</tbody>
</table>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>points <span class="classifier-delimiter">:</span> <span class="classifier">array_like of shape (n,)</span></dt>
<dd>1-D array of monotonically increasing data point coordinates where n
is the number of points. These represent the independent variable values
at which the dependent values are known.</dd>
<dt>values <span class="classifier-delimiter">:</span> <span class="classifier">array_like of shape (n,) or (n, m)</span></dt>
<dd>1-D or 2-D array of data values at the corresponding points. If 1-D,
represents scalar values at each point. If 2-D with shape (n, m),
each column represents a different output dimension, allowing the
lookup table to return m-dimensional vectors.</dd>
<dt>fill_value <span class="classifier-delimiter">:</span> <span class="classifier">float or str, optional</span></dt>
<dd>The value to use for points outside the interpolation range. If &quot;extrapolate&quot;,
the interpolator will use linear extrapolation. Default is &quot;extrapolate&quot;.
See <a class="reference external" href="https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html">https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html</a> for more details</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>points <span class="classifier-delimiter">:</span> <span class="classifier">ndarray</span></dt>
<dd>Flattened array of input point coordinates, stored as 1-D array.</dd>
<dt>values <span class="classifier-delimiter">:</span> <span class="classifier">ndarray</span></dt>
<dd>Stored array of output values at each point, preserving original shape.</dd>
<dt>inter <span class="classifier-delimiter">:</span> <span class="classifier">scipy.interpolate.interp1d</span></dt>
<dd>The scipy 1D interpolator object used for linear interpolation with
extrapolation enabled beyond the data range.</dd>
</dl>
</div>
`,params:{points:{type:"any",default:null,description:"1-D array of monotonically increasing data point coordinates where n is the number of points. These represent the independent variable values at which the dependent values are known."},values:{type:"any",default:null,description:"1-D or 2-D array of data values at the corresponding points. If 1-D, represents scalar values at each point. If 2-D with shape (n, m), each column represents a different output dimension, allowing the lookup table to return m-dimensional vectors."},fill_value:{type:"string",default:'"extrapolate"',description:'The value to use for points outside the interpolation range. If "extrapolate", the interpolator will use linear extrapolation. Default is "extrapolate". See https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html for more details'}},inputs:null,outputs:null},GreaterThan:{blockClass:"GreaterThan",description:"Greater-than comparison block.",docstringHtml:`<p>Greater-than comparison block.</p>
<p>Compares two inputs and outputs 1.0 if a &gt; b, else 0.0.</p>
<div class="math">
\\begin{equation*}
y =
\\begin{cases}
1 &amp; , a &gt; b \\\\
0 &amp; , a \\leq b
\\end{cases}
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:["a","b"],outputs:["y"]},LessThan:{blockClass:"LessThan",description:"Less-than comparison block.",docstringHtml:`<p>Less-than comparison block.</p>
<p>Compares two inputs and outputs 1.0 if a &lt; b, else 0.0.</p>
<div class="math">
\\begin{equation*}
y =
\\begin{cases}
1 &amp; , a &lt; b \\\\
0 &amp; , a \\geq b
\\end{cases}
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:["a","b"],outputs:["y"]},Equal:{blockClass:"Equal",description:"Equality comparison block.",docstringHtml:`<p>Equality comparison block.</p>
<p>Compares two inputs and outputs 1.0 if |a - b| &lt;= tolerance, else 0.0.</p>
<div class="math">
\\begin{equation*}
y =
\\begin{cases}
1 &amp; , |a - b| \\leq \\epsilon \\\\
0 &amp; , |a - b| &gt; \\epsilon
\\end{cases}
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>tolerance <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>comparison tolerance for floating point equality</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{tolerance:{type:"any",default:null,description:"comparison tolerance for floating point equality"}},inputs:["a","b"],outputs:["y"]},LogicAnd:{blockClass:"LogicAnd",description:"Logical AND block.",docstringHtml:`<p>Logical AND block.</p>
<p>Outputs 1.0 if both inputs are nonzero, else 0.0.</p>
<div class="math">
\\begin{equation*}
y = a \\land b
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:["a","b"],outputs:["y"]},LogicOr:{blockClass:"LogicOr",description:"Logical OR block.",docstringHtml:`<p>Logical OR block.</p>
<p>Outputs 1.0 if either input is nonzero, else 0.0.</p>
<div class="math">
\\begin{equation*}
y = a \\lor b
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:["a","b"],outputs:["y"]},LogicNot:{blockClass:"LogicNot",description:"Logical NOT block.",docstringHtml:`<p>Logical NOT block.</p>
<p>Outputs 1.0 if input is zero, else 0.0.</p>
<div class="math">
\\begin{equation*}
y = \\lnot x
\\end{equation*}
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>op_alg <span class="classifier-delimiter">:</span> <span class="classifier">Operator</span></dt>
<dd>internal algebraic operator</dd>
</dl>
</div>
`,params:{},inputs:null,outputs:null},SampleHold:{blockClass:"SampleHold",description:"Zero-order hold: samples the input periodically and holds it at the output.",docstringHtml:`<p>Zero-order hold: samples the input periodically and holds it at the output.</p>
<div class="math">
\\begin{equation*}
y(t) = u(k T + \\tau), \\quad k T + \\tau \\leq t &lt; (k+1) T + \\tau
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - each channel is sampled independently.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic sampling</dd>
</dl>
</div>
`,params:{T:{type:"any",default:null,description:"sampling period"},tau:{type:"any",default:null,description:"delay before first sample"}},inputs:null,outputs:null},ZeroOrderHold:{blockClass:"ZeroOrderHold",description:"Zero-order hold: samples the input periodically and holds it at the output.",docstringHtml:`<p>Zero-order hold: samples the input periodically and holds it at the output.</p>
<div class="math">
\\begin{equation*}
y(t) = u(k T + \\tau), \\quad k T + \\tau \\leq t &lt; (k+1) T + \\tau
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - each channel is sampled independently.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic sampling</dd>
</dl>
</div>
`,params:{T:{type:"any",default:null,description:"sampling period"},tau:{type:"any",default:null,description:"delay before first sample"}},inputs:null,outputs:null},FirstOrderHold:{blockClass:"FirstOrderHold",description:"First-order hold reconstructor.",docstringHtml:`<p>First-order hold reconstructor.</p>
<p>Reconstructs a continuous signal from periodic samples using linear
extrapolation across one sampling interval. Causal (one-sample-lag)
variant matching the Simulink <tt class="docutils literal"><span class="pre">First-Order</span> Hold</tt> block.</p>
<p>Between two consecutive sample times <span class="math">\\(t_{k-1}\\)</span> and <span class="math">\\(t_k\\)</span>,
the output is</p>
<div class="math">
\\begin{equation*}
y(t) = u_{k-1} + \\frac{u_{k-1} - u_{k-2}}{T} (t - t_{k-1})
\\end{equation*}
</div>
<p>During the very first interval (only one sample captured) the output
is held at the most recent sample.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - each channel is extrapolated independently.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic sampling</dd>
</dl>
</div>
`,params:{T:{type:"any",default:null,description:"sampling period"},tau:{type:"any",default:null,description:"delay before first sample"}},inputs:null,outputs:null},FIR:{blockClass:"FIR",description:"Discrete-time Finite-Impulse-Response (FIR) filter.",docstringHtml:`<p>Discrete-time Finite-Impulse-Response (FIR) filter.</p>
<p>Applies an FIR filter to a periodically sampled input signal.</p>
<div class="math">
\\begin{equation*}
y[n] = b_0 x[n] + b_1 x[n-1] + \\dots + b_N x[n-N]
\\end{equation*}
</div>
<p>where <tt class="docutils literal">b</tt> are the filter coefficients and <tt class="docutils literal">N</tt> is the filter order
(number of coefficients minus one). The output is held constant
between sample times.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - the same coefficients are applied to each
channel in parallel.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>coeffs <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>FIR filter coefficients <tt class="docutils literal">[b0, b1, <span class="pre">...,</span> bN]</tt></dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic filter evaluation</dd>
</dl>
</div>
`,params:{coeffs:{type:"any",default:null,description:"FIR filter coefficients ``[b0, b1, ..., bN]``"},T:{type:"number",default:"1.0",description:"sampling period"},tau:{type:"number",default:"0.0",description:"delay before first sample"}},inputs:null,outputs:null},DiscreteIntegrator:{blockClass:"DiscreteIntegrator",description:"Discrete-time integrator (forward Euler).",docstringHtml:`<p>Discrete-time integrator (forward Euler).</p>
<div class="math">
\\begin{equation*}
y[k+1] = y[k] + T \\, u[k]
\\end{equation*}
</div>
<p>The output at sample <tt class="docutils literal">k</tt> is the accumulated sum of past inputs;
the current input <tt class="docutils literal">u[k]</tt> only enters the next sample.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - each channel is integrated independently.
Pass an array as <tt class="docutils literal">initial_value</tt> to set per-channel initial values.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">float, array_like</span></dt>
<dd>initial integrator output <tt class="docutils literal">y[0]</tt></dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic update</dd>
</dl>
</div>
`,params:{T:{type:"number",default:"1.0",description:"sampling period"},tau:{type:"number",default:"0.0",description:"delay before first sample"},initial_value:{type:"any",default:null,description:"initial integrator output ``y[0]``"}},inputs:null,outputs:null},DiscreteDerivative:{blockClass:"DiscreteDerivative",description:"Discrete-time backward-difference derivative.",docstringHtml:`<p>Discrete-time backward-difference derivative.</p>
<div class="math">
\\begin{equation*}
y[k] = \\frac{u[k] - u[k-1]}{T}
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>Supports vector input - each channel is differentiated independently.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic update</dd>
</dl>
</div>
`,params:{T:{type:"any",default:null,description:"sampling period"},tau:{type:"any",default:null,description:"delay before first sample"}},inputs:null,outputs:null},DiscreteStateSpace:{blockClass:"DiscreteStateSpace",description:"Discrete-time MIMO state space block.",docstringHtml:`<p>Discrete-time MIMO state space block.</p>
<div class="math">
\\begin{equation*}
\\begin{align}
    x[k+1] &amp;= \\mathbf{A}\\, x[k] + \\mathbf{B}\\, u[k] \\\\
    y[k]   &amp;= \\mathbf{C}\\, x[k] + \\mathbf{D}\\, u[k]
\\end{align}
\\end{equation*}
</div>
<div class="section" id="note">
<h3>Note</h3>
<p>The output port reflects <tt class="docutils literal">y[k]</tt> for the duration of the current
sample interval (zero-order hold between updates). The direct
feedthrough term <tt class="docutils literal">D u[k]</tt> is computed at the sample event, so the
block has no algebraic passthrough between updates.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>A, B, C, D <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>discrete state space matrices</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
<dt>initial_value <span class="classifier-delimiter">:</span> <span class="classifier">array_like, None</span></dt>
<dd>initial state <tt class="docutils literal">x[0]</tt></dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic update</dd>
</dl>
</div>
`,params:{A:{type:"any",default:null,description:""},B:{type:"any",default:null,description:""},C:{type:"any",default:null,description:""},D:{type:"any",default:null,description:"discrete state space matrices"},T:{type:"number",default:"1.0",description:"sampling period"},tau:{type:"number",default:"0.0",description:"delay before first sample"},initial_value:{type:"any",default:null,description:"initial state ``x[0]``"}},inputs:null,outputs:null},DiscreteTransferFunction:{blockClass:"DiscreteTransferFunction",description:"Discrete-time SISO transfer function in numerator/denominator form.",docstringHtml:`<p>Discrete-time SISO transfer function in numerator/denominator form.</p>
<div class="math">
\\begin{equation*}
H(z) = \\frac{b_0 z^M + b_1 z^{M-1} + \\dots + b_M}{a_0 z^N + a_1 z^{N-1} + \\dots + a_N}
\\end{equation*}
</div>
<p>Realized internally as a <tt class="docutils literal">DiscreteStateSpace</tt> via the controllable
canonical form returned by <tt class="docutils literal">scipy.signal.tf2ss</tt>.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>Num <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>numerator polynomial coefficients (highest power of z first)</dd>
<dt>Den <span class="classifier-delimiter">:</span> <span class="classifier">array_like</span></dt>
<dd>denominator polynomial coefficients (highest power of z first)</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
`,params:{Num:{type:"any",default:null,description:"numerator polynomial coefficients (highest power of z first)"},Den:{type:"any",default:null,description:"denominator polynomial coefficients (highest power of z first)"},T:{type:"number",default:"1.0",description:"sampling period"},tau:{type:"number",default:"0.0",description:"delay before first sample"}},inputs:["in"],outputs:["out"]},TappedDelay:{blockClass:"TappedDelay",description:"Tapped delay line.",docstringHtml:`<p>Tapped delay line.</p>
<p>Outputs the current and <tt class="docutils literal"><span class="pre">N-1</span></tt> past samples of the input as parallel
signals. The block has <tt class="docutils literal">N</tt> outputs:</p>
<div class="math">
\\begin{equation*}
y_i[k] = u[k - i], \\quad i = 0, 1, \\dots, N-1
\\end{equation*}
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>N <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>number of taps (output ports)</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay before first sample</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>internal scheduled event for periodic shift</dd>
</dl>
</div>
`,params:{N:{type:"integer",default:"2",description:"number of taps (output ports)"},T:{type:"number",default:"1.0",description:"sampling period"},tau:{type:"number",default:"0.0",description:"delay before first sample"}},inputs:["in"],outputs:null},ADC:{blockClass:"ADC",description:"Models an ideal Analog-to-Digital Converter (ADC).",docstringHtml:`<p>Models an ideal Analog-to-Digital Converter (ADC).</p>
<p>This block samples an analog input signal periodically, quantizes it
according to the specified number of bits and input span, and outputs
the resulting digital code on multiple output ports. The sampling
is triggered by a scheduled event.</p>
<p>Functionality:</p>
<ol class="arabic simple">
<li>Samples the analog input <cite>inputs[0]</cite> at intervals of <cite>T</cite>, starting after delay <cite>tau</cite>.</li>
<li>Clips the input voltage to the defined <cite>span</cite> [min_voltage, max_voltage].</li>
<li>Scales the clipped voltage to the range [0, 1].</li>
<li>Quantizes the scaled value to an integer code between 0 and 2^n_bits - 1 using flooring.</li>
<li>Converts the integer code to an n_bits binary representation.</li>
<li>Outputs the binary code on ports 0 (LSB) to n_bits-1 (MSB).</li>
</ol>
<p>Ideal characteristics:</p>
<ul class="simple">
<li>Instantaneous sampling at scheduled times.</li>
<li>Perfect, noise-free quantization.</li>
<li>No aperture jitter or other dynamic errors.</li>
</ul>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>n_bits <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span></dt>
<dd>Number of bits for the digital output code. Default is 4.</dd>
<dt>span <span class="classifier-delimiter">:</span> <span class="classifier">list[float] or tuple[float], optional</span></dt>
<dd>The valid analog input value range [min_voltage, max_voltage].
Inputs outside this range will be clipped. Default is [-1, 1].</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Sampling period (time between samples). Default is 1 time unit.</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Initial delay before the first sample is taken. Default is 0.</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>Internal scheduled event responsible for periodic sampling and conversion.</dd>
</dl>
</div>
`,params:{n_bits:{type:"integer",default:"4",description:"Number of bits for the digital output code. Default is 4."},span:{type:"any",default:null,description:"The valid analog input value range [min_voltage, max_voltage]. Inputs outside this range will be clipped. Default is [-1, 1]."},T:{type:"number",default:"1.0",description:"Sampling period (time between samples). Default is 1 time unit."},tau:{type:"number",default:"0.0",description:"Initial delay before the first sample is taken. Default is 0."}},inputs:["in"],outputs:null},DAC:{blockClass:"DAC",description:"Models an ideal Digital-to-Analog Converter (DAC).",docstringHtml:`<p>Models an ideal Digital-to-Analog Converter (DAC).</p>
<p>This block reads a digital input code periodically from its input ports,
reconstructs the corresponding analog value based on the number of bits
and output span, and holds the output constant between updates. The update
is triggered by a scheduled event.</p>
<p>Functionality:</p>
<ol class="arabic simple">
<li>Reads the digital code from input ports 0 (LSB) to n_bits-1 (MSB) at intervals of <cite>T</cite>, starting after delay <cite>tau</cite>.</li>
<li>Interprets the inputs as an unsigned binary integer code.</li>
<li>Converts the integer code to a fractional value between 0 and (2^n_bits - 1) / 2^n_bits.</li>
<li>Scales this fractional value to the specified analog output <cite>span</cite>.</li>
<li>Outputs the resulting analog value on <cite>outputs[0]</cite>.</li>
<li>Holds the output value constant until the next scheduled update.</li>
</ol>
<p>Ideal characteristics:</p>
<ul class="simple">
<li>Instantaneous update at scheduled times.</li>
<li>Perfect, noise-free reconstruction.</li>
<li>No glitches or settling time.</li>
</ul>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>n_bits <span class="classifier-delimiter">:</span> <span class="classifier">int, optional</span></dt>
<dd>Number of digital input bits expected. Default is 4.</dd>
<dt>span <span class="classifier-delimiter">:</span> <span class="classifier">list[float] or tuple[float], optional</span></dt>
<dd>The analog output value range [min_voltage, max_voltage] corresponding
to the digital codes 0 and 2^n_bits - 1, respectively (approximately).
Default is [-1, 1].</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Update period (time between output updates). Default is 1 time unit.</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float, optional</span></dt>
<dd>Initial delay before the first output update. Default is 0.</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[Schedule]</span></dt>
<dd>Internal scheduled event responsible for periodic updates.</dd>
</dl>
</div>
`,params:{n_bits:{type:"integer",default:"4",description:"Number of digital input bits expected. Default is 4."},span:{type:"any",default:null,description:"The analog output value range [min_voltage, max_voltage] corresponding to the digital codes 0 and 2^n_bits - 1, respectively (approximately). Default is [-1, 1]."},T:{type:"number",default:"1.0",description:"Update period (time between output updates). Default is 1 time unit."},tau:{type:"number",default:"0.0",description:"Initial delay before the first output update. Default is 0."}},inputs:null,outputs:["out"]},Counter:{blockClass:"Counter",description:"Counts the number of detected bidirectional threshold crossings.",docstringHtml:`<p>Counts the number of detected bidirectional threshold crossings.</p>
<p>Uses zero-crossing events for the detection and sets the output
accordingly.</p>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>start <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>counter start (initial condition)</dd>
<dt>threshold <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>threshold for zero crossing</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>E <span class="classifier-delimiter">:</span> <span class="classifier">ZeroCrossing</span></dt>
<dd>internal event manager</dd>
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[ZeroCrossing]</span></dt>
<dd>internal zero crossing event</dd>
</dl>
</div>
`,params:{start:{type:"any",default:null,description:"counter start (initial condition)"},threshold:{type:"any",default:null,description:"threshold for zero crossing"}},inputs:["in"],outputs:["out"]},CounterUp:{blockClass:"CounterUp",description:"Counts the number of detected unidirectional (lo->hi) threshold crossings.",docstringHtml:`<p>Counts the number of detected unidirectional (lo-&gt;hi) threshold crossings.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>This is a modification of 'Counter' which only counts
unidirectional zero-crossings (low -&gt; high)</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>start <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>counter start (initial condition)</dd>
<dt>threshold <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>threshold for zero crossing</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>E <span class="classifier-delimiter">:</span> <span class="classifier">ZeroCrossingUp</span></dt>
<dd>internal event manager</dd>
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[ZeroCrossing]</span></dt>
<dd>internal zero crossing event</dd>
</dl>
</div>
`,params:{start:{type:"any",default:null,description:"counter start (initial condition)"},threshold:{type:"any",default:null,description:"threshold for zero crossing"}},inputs:["in"],outputs:["out"]},CounterDown:{blockClass:"CounterDown",description:"Counts the number of detected unidirectional (hi->lo) threshold crossings.",docstringHtml:`<p>Counts the number of detected unidirectional (hi-&gt;lo) threshold crossings.</p>
<div class="section" id="note">
<h3>Note</h3>
<p>This is a modification of 'Counter' which only counts
unidirectional zero-crossings (high -&gt; low)</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>start <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>counter start (initial condition)</dd>
<dt>threshold <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>threshold for zero crossing</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>E <span class="classifier-delimiter">:</span> <span class="classifier">ZeroCrossingDown</span></dt>
<dd>internal event manager</dd>
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[ZeroCrossing]</span></dt>
<dd>internal zero crossing event</dd>
</dl>
</div>
`,params:{start:{type:"any",default:null,description:"counter start (initial condition)"},threshold:{type:"any",default:null,description:"threshold for zero crossing"}},inputs:["in"],outputs:["out"]},Relay:{blockClass:"Relay",description:"Relay block with hysteresis (Schmitt trigger).",docstringHtml:`<p>Relay block with hysteresis (Schmitt trigger).</p>
<p>Switches output between two values based on input crossing upper and lower
thresholds. The hysteresis prevents rapid switching when input is noisy.</p>
<p>When input rises above <cite>threshold_up</cite>, output switches to <cite>value_up</cite>.
When input falls below <cite>threshold_down</cite>, output switches to <cite>value_down</cite>.</p>
<div class="section" id="examples">
<h3>Examples</h3>
<p>Basic thermostat that turns heater on below 19°C, off above 21°C:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Relay</span><span class="whitespace">

</span><span class="name">thermostat</span> <span class="operator">=</span> <span class="name">Relay</span><span class="punctuation">(</span><span class="whitespace">
</span>    <span class="name">threshold_up</span><span class="operator">=</span><span class="literal number float">21.0</span><span class="punctuation">,</span><span class="whitespace">
</span>    <span class="name">threshold_down</span><span class="operator">=</span><span class="literal number float">19.0</span><span class="punctuation">,</span><span class="whitespace">
</span>    <span class="name">value_up</span><span class="operator">=</span><span class="literal number float">0.0</span><span class="punctuation">,</span><span class="whitespace">
</span>    <span class="name">value_down</span><span class="operator">=</span><span class="literal number float">1.0</span><span class="whitespace">
</span>    <span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>threshold_up <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>threshold for transitioning to upper relay state <cite>value_up</cite> (default: 1.0)</dd>
<dt>threshold_down <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>threshold for transitioning to lower relay state <cite>value_down</cite> (default: 0.0)</dd>
<dt>value_up <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>value for upper relay state (default: 1.0)</dd>
<dt>value_down <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>value for lower relay state (default: 0.0)</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>events <span class="classifier-delimiter">:</span> <span class="classifier">list[ZeroCrossingUp, ZeroCrossingDown]</span></dt>
<dd>internal zero crossing events for relay state transitions</dd>
</dl>
</div>
`,params:{threshold_up:{type:"any",default:null,description:"threshold for transitioning to upper relay state `value_up` (default: 1.0)"},threshold_down:{type:"any",default:null,description:"threshold for transitioning to lower relay state `value_down` (default: 0.0)"},value_up:{type:"any",default:null,description:"value for upper relay state (default: 1.0)"},value_down:{type:"any",default:null,description:"value for lower relay state (default: 0.0)"}},inputs:["in"],outputs:["out"]},Wrapper:{blockClass:"Wrapper",description:"Wrapper block for discrete implementation and external code integration.",docstringHtml:`<p>Wrapper block for discrete implementation and external code integration.</p>
<p>The <cite>Wrapper</cite> class is designed to call the internal <cite>func</cite> at fixed intervals
using an internal <cite>Schedule</cite> event. This makes it particularly useful for wrapping
external code or implementing discrete-time systems.</p>
<p>Essentially this block does the same as <cite>Function</cite> with the difference that its
not evaluated continuously but periodically at discrete times.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>There are two ways to setup the <cite>Wrapper</cite>, first and standard way is to define
a function to be wrapped and pass it to the block initializer:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Wrapper</span><span class="whitespace">

</span><span class="comment single">#function to be wrapped</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">func</span><span class="punctuation">(</span><span class="name">a</span><span class="punctuation">,</span> <span class="name">b</span><span class="punctuation">,</span> <span class="name">c</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">a</span> <span class="operator">*</span> <span class="punctuation">(</span><span class="name">b</span> <span class="operator">+</span> <span class="name">c</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="name">wrp</span> <span class="operator">=</span> <span class="name">Wrapper</span><span class="punctuation">(</span><span class="name">func</span><span class="punctuation">,</span> <span class="name">T</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">)</span>
</pre>
<p>Another option is to use the <cite>dec</cite> classmethod, which might be more convenient
in some situations:</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Wrapper</span><span class="whitespace">

</span><span class="name decorator">&#64;Wrapper</span><span class="operator">.</span><span class="name">dec</span><span class="punctuation">(</span><span class="name">T</span><span class="operator">=</span><span class="literal number float">0.1</span><span class="punctuation">)</span><span class="whitespace">
</span><span class="keyword">def</span> <span class="name function">wrp</span><span class="punctuation">(</span><span class="name">a</span><span class="punctuation">,</span> <span class="name">b</span><span class="punctuation">,</span> <span class="name">c</span><span class="punctuation">):</span><span class="whitespace">
</span>    <span class="keyword">return</span> <span class="name">a</span> <span class="operator">*</span> <span class="punctuation">(</span><span class="name">b</span> <span class="operator">+</span> <span class="name">c</span><span class="punctuation">)</span>
</pre>
<p>This way the internal function of the block <cite>wrp</cite> will be evaluated with a period
of <cite>T=0.1</cite> and its outputs updated accordingly.</p>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>func <span class="classifier-delimiter">:</span> <span class="classifier">callable</span></dt>
<dd>function that defines algebraic block IO behaviour</dd>
<dt>T <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>sampling period for the wrapped function</dd>
<dt>tau <span class="classifier-delimiter">:</span> <span class="classifier">float</span></dt>
<dd>delay time for the start time of the event</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>Evt <span class="classifier-delimiter">:</span> <span class="classifier">Schedule</span></dt>
<dd>internal event. Used for periodic sampling the wrapped method</dd>
</dl>
</div>
`,params:{func:{type:"callable",default:null,description:"function that defines algebraic block IO behaviour"},T:{type:"number",default:"1.0",description:"sampling period for the wrapped function"},tau:{type:"number",default:"0.0",description:"delay time for the start time of the event Attributes ----------"}},inputs:null,outputs:null},ModelExchangeFMU:{blockClass:"ModelExchangeFMU",description:"ModelExchangeFMU block",docstringHtml:`<p>ModelExchangeFMU block</p>
`,params:{fmu_path:{type:"any",default:null,description:""},instance_name:{type:"string",default:'"fmu_instance"',description:""},start_values:{type:"any",default:null,description:""},tolerance:{type:"number",default:"1e-10",description:""},verbose:{type:"boolean",default:"false",description:""}},inputs:null,outputs:null},CoSimulationFMU:{blockClass:"CoSimulationFMU",description:"CoSimulationFMU block",docstringHtml:`<p>CoSimulationFMU block</p>
`,params:{fmu_path:{type:"any",default:null,description:""},instance_name:{type:"string",default:'"fmu_instance"',description:""},start_values:{type:"any",default:null,description:""},dt:{type:"any",default:null,description:""},verbose:{type:"boolean",default:"false",description:""}},inputs:null,outputs:null},Scope:{blockClass:"Scope",description:"Scope block with plot() method for quick visualization.",docstringHtml:`<p>Scope block with plot() method for quick visualization.</p>
`,params:{labels:{type:"any",default:null,description:""},sampling_period:{type:"any",default:null,description:""},t_wait:{type:"number",default:"0.0",description:""}},inputs:[],outputs:[]},Spectrum:{blockClass:"Spectrum",description:"Spectrum block with plot() method for frequency-domain visualization.",docstringHtml:`<p>Spectrum block with plot() method for frequency-domain visualization.</p>
`,params:{freq:{type:"any",default:null,description:""},t_wait:{type:"number",default:"0.0",description:""},alpha:{type:"number",default:"0.0",description:""},labels:{type:"any",default:null,description:""}},inputs:[],outputs:[]},Subsystem:{blockClass:"Subsystem",description:"Subsystem class that holds its own blocks and connecions and",docstringHtml:`<p>Subsystem class that holds its own blocks and connecions and
can natively interface with the main simulation loop.</p>
<p>IO interface is realized by a special 'Interface' block, that has extra
methods for setting and getting inputs and outputs and serves
as the interface of the internal blocks to the outside.</p>
<p>The subsystem doesnt use its 'inputs' and 'outputs' dicts directly.
It exclusively handles data transfer via the 'Interface' block.</p>
<p>This class can be used just like any other block during the simulation,
since it implements the required methods 'update' for the fixed-point
iteration (resolving algebraic loops with instant time blocks),
the 'step' method that performs timestepping (especially for dynamic
blocks with internal states) and the 'solve' method for solving the
implicit update equation for implicit solvers.</p>
<div class="section" id="example">
<h3>Example</h3>
<p>This is how we can wrap up multiple blocks within a subsystem.
In this case vanderpol system built from discrete components
instead of using an ODE block (in practice you should use
a monolithic ODE whenever possible due to performance).</p>
<pre class="code python literal-block">
<span class="keyword namespace">from</span> <span class="name namespace">pathsim</span> <span class="keyword namespace">import</span> <span class="name">Subsystem</span><span class="punctuation">,</span> <span class="name">Interface</span><span class="punctuation">,</span> <span class="name">Connection</span><span class="whitespace">
</span><span class="keyword namespace">from</span> <span class="name namespace">pathsim.blocks</span> <span class="keyword namespace">import</span> <span class="name">Integrator</span><span class="punctuation">,</span> <span class="name">Function</span><span class="whitespace">

</span><span class="comment single">#van der Pol parameter</span><span class="whitespace">
</span><span class="name">mu</span> <span class="operator">=</span> <span class="literal number integer">1000</span><span class="whitespace">

</span><span class="comment single">#blocks in the subsystem</span><span class="whitespace">
</span><span class="name">If</span> <span class="operator">=</span> <span class="name">Interface</span><span class="punctuation">()</span> <span class="comment single"># this is the interface to the outside</span><span class="whitespace">
</span><span class="name">I1</span> <span class="operator">=</span> <span class="name">Integrator</span><span class="punctuation">(</span><span class="literal number integer">2</span><span class="punctuation">)</span><span class="whitespace">
</span><span class="name">I2</span> <span class="operator">=</span> <span class="name">Integrator</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">)</span><span class="whitespace">
</span><span class="name">Fn</span> <span class="operator">=</span> <span class="name">Function</span><span class="punctuation">(</span><span class="keyword">lambda</span> <span class="name">x1</span><span class="punctuation">,</span> <span class="name">x2</span><span class="punctuation">:</span> <span class="name">mu</span><span class="operator">*</span><span class="punctuation">(</span><span class="literal number integer">1</span> <span class="operator">-</span> <span class="name">x1</span><span class="operator">**</span><span class="literal number integer">2</span><span class="punctuation">)</span><span class="operator">*</span><span class="name">x2</span> <span class="operator">-</span> <span class="name">x1</span><span class="punctuation">)</span><span class="whitespace">

</span><span class="name">sub_blocks</span> <span class="operator">=</span> <span class="punctuation">[</span><span class="name">If</span><span class="punctuation">,</span> <span class="name">I1</span><span class="punctuation">,</span> <span class="name">I2</span><span class="punctuation">,</span> <span class="name">Fn</span><span class="punctuation">]</span><span class="whitespace">

</span><span class="comment single">#connections in the subsystem</span><span class="whitespace">
</span><span class="name">sub_connections</span> <span class="operator">=</span> <span class="punctuation">[</span><span class="whitespace">
</span>    <span class="name">Connection</span><span class="punctuation">(</span><span class="name">I2</span><span class="punctuation">,</span> <span class="name">I1</span><span class="punctuation">,</span> <span class="name">Fn</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">],</span> <span class="name">If</span><span class="punctuation">[</span><span class="literal number integer">1</span><span class="punctuation">]),</span><span class="whitespace">
</span>    <span class="name">Connection</span><span class="punctuation">(</span><span class="name">I1</span><span class="punctuation">,</span> <span class="name">Fn</span><span class="punctuation">,</span> <span class="name">If</span><span class="punctuation">),</span><span class="whitespace">
</span>    <span class="name">Connection</span><span class="punctuation">(</span><span class="name">Fn</span><span class="punctuation">,</span> <span class="name">I2</span><span class="punctuation">)</span><span class="whitespace">
</span>    <span class="punctuation">]</span><span class="whitespace">

</span><span class="comment single">#the subsystem acts just like a normal block</span><span class="whitespace">
</span><span class="name">vdp</span> <span class="operator">=</span> <span class="name">Subsystem</span><span class="punctuation">(</span><span class="name">sub_blocks</span><span class="punctuation">,</span> <span class="name">sub_connections</span><span class="punctuation">)</span>
</pre>
</div>
<div class="section" id="parameters">
<h3>Parameters</h3>
<dl class="docutils">
<dt>blocks <span class="classifier-delimiter">:</span> <span class="classifier">list[Block] | None</span></dt>
<dd>internal blocks of the subsystem</dd>
<dt>connections <span class="classifier-delimiter">:</span> <span class="classifier">list[Connection] | None</span></dt>
<dd>internal connections of the subsystem</dd>
</dl>
<p>events : list[Event] | None
tolerance_fpi : float</p>
<blockquote>
absolute tolerance for convergence of algebraic loops
default see ´SIM_TOLERANCE_FPI´ in ´_constants.py´</blockquote>
<dl class="docutils">
<dt>iterations_max <span class="classifier-delimiter">:</span> <span class="classifier">int</span></dt>
<dd>maximum allowed number of iterations for algebraic loop
solver, default see ´SIM_ITERATIONS_MAX´ in ´_constants.py´</dd>
</dl>
</div>
<div class="section" id="attributes">
<h3>Attributes</h3>
<dl class="docutils">
<dt>interface <span class="classifier-delimiter">:</span> <span class="classifier">Interface</span></dt>
<dd>internal interface block for data transfer to the outside</dd>
<dt>graph <span class="classifier-delimiter">:</span> <span class="classifier">Graph</span></dt>
<dd>internal graph representation for fast system funcion
evluations using DAG with algebraic depths</dd>
<dt>boosters <span class="classifier-delimiter">:</span> <span class="classifier">None | list[ConnectionBooster]</span></dt>
<dd>list of boosters (fixed point accelerators) that wrap
algebraic loop closing connections assembled from the
system graph</dd>
</dl>
</div>
`,params:{},inputs:[],outputs:[]},Interface:{blockClass:"Interface",description:"Bare-bone block that serves as a data interface for the 'Subsystem' class.",docstringHtml:`<p>Bare-bone block that serves as a data interface for the 'Subsystem' class.</p>
<p>It works like this:</p>
<ul class="simple">
<li>Internal blocks of the subsystem are connected to the inputs and outputs
of this Interface block via the internal connections.</li>
<li>It behaves like a normal block (inherits the main 'Block' class methods).</li>
<li>It implements some special methods to get and set the inputs and outputs
of the blocks, that are used to translate between the internal blocks of the
subsystem and the inputs and outputs of the subsystem.</li>
<li>Handles data transfer to and from the internal subsystem blocks
to and from the inputs and outputs of the subsystem.</li>
</ul>
`,params:{},inputs:[],outputs:[]}},na={Sources:["Constant","Source","SinusoidalSource","StepSource","PulseSource","TriangleWaveSource","SquareWaveSource","GaussianPulseSource","ChirpPhaseNoiseSource","ClockSource","WhiteNoise","PinkNoise","RandomNumberGenerator"],Dynamic:["Integrator","Differentiator","Delay","ODE","DynamicalSystem","StateSpace","PT1","PT2","LeadLag","PID","AntiWindupPID","RateLimiter","Backlash","Deadband","TransferFunctionNumDen","TransferFunctionZPG","ButterworthLowpassFilter","ButterworthHighpassFilter","ButterworthBandpassFilter","ButterworthBandstopFilter"],DAE:["MassMatrixDAE","SemiExplicitDAE","FullyImplicitDAE"],Algebraic:["Adder","Multiplier","Divider","Amplifier","Function","Polynomial","Sin","Cos","Tan","Tanh","Abs","Sqrt","Exp","Log","Log10","Mod","Clip","Pow","Atan2","Rescale","Alias","Switch","LUT1D"],Logic:["GreaterThan","LessThan","Equal","LogicAnd","LogicOr","LogicNot"],Discrete:["SampleHold","ZeroOrderHold","FirstOrderHold","FIR","DiscreteIntegrator","DiscreteDerivative","DiscreteStateSpace","DiscreteTransferFunction","TappedDelay","ADC","DAC","Counter","CounterUp","CounterDown","Relay","Wrapper"],FMI:["ModelExchangeFMU","CoSimulationFMU"],Recording:["Scope","Spectrum"]},ge={ADC:"fastsim.blocks",Abs:"fastsim.blocks",Adder:"fastsim.blocks",Alias:"fastsim.blocks",Amplifier:"fastsim.blocks",AntiWindupPID:"fastsim.blocks",Atan2:"fastsim.blocks",Backlash:"fastsim.blocks",ButterworthBandpassFilter:"fastsim.blocks",ButterworthBandstopFilter:"fastsim.blocks",ButterworthHighpassFilter:"fastsim.blocks",ButterworthLowpassFilter:"fastsim.blocks",ChirpPhaseNoiseSource:"fastsim.blocks",Clip:"fastsim.blocks",ClockSource:"fastsim.blocks",CoSimulationFMU:"fastsim.blocks",Constant:"fastsim.blocks",Cos:"fastsim.blocks",Counter:"fastsim.blocks",CounterDown:"fastsim.blocks",CounterUp:"fastsim.blocks",DAC:"fastsim.blocks",Deadband:"fastsim.blocks",Delay:"fastsim.blocks",Differentiator:"fastsim.blocks",DiscreteDerivative:"fastsim.blocks",DiscreteIntegrator:"fastsim.blocks",DiscreteStateSpace:"fastsim.blocks",DiscreteTransferFunction:"fastsim.blocks",Divider:"fastsim.blocks",DynamicalSystem:"fastsim.blocks",Equal:"fastsim.blocks",Exp:"fastsim.blocks",FIR:"fastsim.blocks",FirstOrderHold:"fastsim.blocks",FullyImplicitDAE:"fastsim.blocks",Function:"fastsim.blocks",GaussianPulseSource:"fastsim.blocks",GreaterThan:"fastsim.blocks",Integrator:"fastsim.blocks",LUT1D:"fastsim.blocks",LeadLag:"fastsim.blocks",LessThan:"fastsim.blocks",Log:"fastsim.blocks",Log10:"fastsim.blocks",LogicAnd:"fastsim.blocks",LogicNot:"fastsim.blocks",LogicOr:"fastsim.blocks",MassMatrixDAE:"fastsim.blocks",Mod:"fastsim.blocks",ModelExchangeFMU:"fastsim.blocks",Multiplier:"fastsim.blocks",ODE:"fastsim.blocks",PID:"fastsim.blocks",PT1:"fastsim.blocks",PT2:"fastsim.blocks",PinkNoise:"fastsim.blocks",Polynomial:"fastsim.blocks",Pow:"fastsim.blocks",PulseSource:"fastsim.blocks",RandomNumberGenerator:"fastsim.blocks",RateLimiter:"fastsim.blocks",Relay:"fastsim.blocks",Rescale:"fastsim.blocks",SampleHold:"fastsim.blocks",Scope:"fastsim.blocks",SemiExplicitDAE:"fastsim.blocks",Sin:"fastsim.blocks",SinusoidalSource:"fastsim.blocks",Source:"fastsim.blocks",Spectrum:"fastsim.blocks",Sqrt:"fastsim.blocks",SquareWaveSource:"fastsim.blocks",StateSpace:"fastsim.blocks",StepSource:"fastsim.blocks",Switch:"fastsim.blocks",Tan:"fastsim.blocks",Tanh:"fastsim.blocks",TappedDelay:"fastsim.blocks",TransferFunctionNumDen:"fastsim.blocks",TransferFunctionZPG:"fastsim.blocks",TriangleWaveSource:"fastsim.blocks",WhiteNoise:"fastsim.blocks",Wrapper:"fastsim.blocks",ZeroOrderHold:"fastsim.blocks"};function Cs(a){if(a==null||a==="None"||a==="")return null;let n=String(a).trim();return n.length===0||((n.startsWith("'")&&n.endsWith("'")||n.startsWith('"')&&n.endsWith('"'))&&(n=n.slice(1,-1)),n.length===0)?null:[...n]}const ea={Scope:{param:"labels",direction:"input"},Spectrum:{param:"labels",direction:"input"},Adder:{param:"operations",direction:"input",parser:Cs},Divider:{param:"operations",direction:"input",parser:Cs}};function be(a){const n=ea[a];return n?Array.isArray(n)?n:[n]:[]}const ta=new Set(["Integrator","Differentiator","Delay","Amplifier","Sin","Cos","Tan","Tanh","Abs","Sqrt","Exp","Log","Log10","Mod","Clip","Pow","Polynomial","Rescale","Alias","LogicNot","SampleHold","FirstOrderHold","DiscreteIntegrator","DiscreteDerivative"]),Ws="builtin";class ia{nodes=new Map;byCategory=new Map;bySource=new Map;register(n,s=Ws){this.nodes.has(n.type)&&(console.warn(`[nodeRegistry] replacing "${n.type}" (was ${this.nodes.get(n.type)?.source}, now ${s})`),this.removeFromIndexes(n.type)),this.nodes.set(n.type,{definition:n,source:s});const e=this.byCategory.get(n.category)??new Set;e.add(n.type),this.byCategory.set(n.category,e);const t=this.bySource.get(s)??new Set;t.add(n.type),this.bySource.set(s,t),As()}unregisterSource(n){const s=Array.from(this.bySource.get(n)??[]);for(const e of s)this.removeFromIndexes(e),this.nodes.delete(e);return this.bySource.delete(n),s.length>0&&As(),s}removeFromIndexes(n){const s=this.nodes.get(n);if(!s)return;const e=this.byCategory.get(s.definition.category);e&&(e.delete(n),e.size===0&&this.byCategory.delete(s.definition.category));const t=this.bySource.get(s.source);t&&(t.delete(n),t.size===0&&this.bySource.delete(s.source))}get(n){return this.nodes.get(n)?.definition}getSource(n){return this.nodes.get(n)?.source}getByCategory(n){const s=this.byCategory.get(n);return s?Array.from(s).map(e=>this.nodes.get(e)?.definition).filter(e=>!!e):[]}getBySource(n){const s=this.bySource.get(n);return s?Array.from(s).map(e=>this.nodes.get(e)?.definition).filter(e=>!!e):[]}getAllCategories(){return Array.from(this.byCategory.keys())}getAllSources(){return Array.from(this.bySource.keys())}getAll(){return Array.from(this.nodes.values()).map(n=>n.definition)}has(n){return this.nodes.has(n)}get size(){return this.nodes.size}}const Ks=Xs(0);function As(){Ks.update(a=>a+1)}const ve={subscribe:Ks.subscribe},la=new ia;function pa(a,n,s){const e={};for(const[u,d]of Object.entries(s.params))e[u]={type:d.type,default:d.default,description:d.description,min:d.min,max:d.max,options:d.options};let t,i;s.inputs===null?(t=void 0,i=null):s.inputs.length>0?(t=s.inputs,i=s.inputs.length):(t=[],i=0);let p,o;s.outputs===null?(p=void 0,o=null):s.outputs.length>0?(p=s.outputs,o=s.outputs.length):(p=[],o=0);const c=sa({name:a,category:n,blockClass:s.blockClass,description:s.description,inputs:t,outputs:p,maxInputs:i,maxOutputs:o,syncPorts:ta.has(a),params:e});s.docstringHtml&&(c.docstring=s.docstringHtml),la.register(c,Ws)}function oa(){for(const[a,n]of Object.entries(na))for(const s of n){const e=aa[s];e?pa(s,a,e):console.warn(`Block "${s}" not found in extracted blocks`)}}oa();const ra=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 62 19 L 62 14 L 34 14 L 44 32 L 34 50 L 62 50 L 62 45"/>
</svg>
`,ca=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="18" y1="32" x2="34" y2="32"/>
  <line x1="62" y1="32" x2="78" y2="32"/>
  <path d="M 34 22 H 56 L 62 32 L 56 42 H 34 Z"/>
  <circle cx="40" cy="32" r="2" fill="currentColor" stroke="none"/>
</svg>
`,da=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <polygon points="27,14 27,50 69,32"/>
</svg>
`,ua=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="32" y1="32" x2="64" y2="32"/>
  <circle cx="48" cy="20" r="3" fill="currentColor" stroke="none"/>
  <circle cx="48" cy="44" r="3" fill="currentColor" stroke="none"/>
</svg>
`,ma=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="32" y1="25" x2="64" y2="25"/>
  <line x1="32" y1="39" x2="64" y2="39"/>
</svg>
`,fa=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 36 17 L 62 32 L 36 47"/>
</svg>
`,ha=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="48" y1="12" x2="48" y2="52" stroke-dasharray="4 3"/>
  <path d="M 24 22 L 42 22 M 36 16 L 42 22 L 36 28"/>
  <path d="M 54 42 L 72 42 M 66 36 L 72 42 L 66 48"/>
</svg>
`,ga=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 60 17 L 34 32 L 60 47"/>
</svg>
`,ba=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 34 14 H 48 A 18 18 0 0 1 48 50 H 34 Z"/>
</svg>
`,va=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <polygon points="30,14 30,50 58,32"/>
  <circle cx="62" cy="32" r="4"/>
</svg>
`,ya=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 30 14 Q 41 32 30 50 Q 54 50 66 32 Q 54 14 30 14 Z"/>
</svg>
`,ka=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 36 50 L 36 14 M 60 50 L 60 14 M 31 14 L 65 14"/>
</svg>
`,wa=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <rect x="6" y="8" width="84" height="48" rx="3" stroke-dasharray="4 3"/>
  <rect x="14" y="20" width="18" height="10" rx="1.5"/>
  <rect x="39" y="34" width="18" height="10" rx="1.5"/>
  <rect x="64" y="20" width="18" height="10" rx="1.5"/>
</svg>
`,_a=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="22" y1="22" x2="40" y2="22"/>
  <line x1="22" y1="42" x2="40" y2="42"/>
  <line x1="56" y1="32" x2="74" y2="32"/>
  <line x1="40" y1="22" x2="56" y2="32"/>
  <circle cx="40" cy="22" r="2.5" fill="currentColor" stroke="none"/>
  <circle cx="40" cy="42" r="2.5" fill="currentColor" stroke="none"/>
  <circle cx="56" cy="32" r="2.5" fill="currentColor" stroke="none"/>
</svg>
`,xa=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <line x1="10" y1="56" x2="88" y2="56"/>
  <line x1="12" y1="8" x2="12" y2="58"/>
  <path d="M 16 18 L 22 18 L 22 13 L 32 13 L 32 18 L 80 18"/>
  <path d="M 16 28 L 34 28 L 34 23 L 44 23 L 44 28 L 80 28"/>
  <path d="M 16 38 L 46 38 L 46 33 L 56 33 L 56 38 L 80 38"/>
  <path d="M 16 48 L 58 48 L 58 43 L 68 43 L 68 48 L 80 48"/>
</svg>
`,Sa=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <rect x="22" y="12" width="52" height="40" rx="4" stroke-dasharray="4 3"/>
  <rect x="36" y="24" width="24" height="16" rx="2"/>
</svg>
`,W={x0:14,x1:82,y0:14,y1:50,get width(){return this.x1-this.x0},get height(){return this.y1-this.y0}},gs=4,Y={x0:W.x0-gs,x1:W.x1+gs,y0:W.y0-gs,y1:W.y1+gs};function us(a,n=0,s=1){const e=(a-n)/(s-n);return W.x0+e*W.width}function ms(a,n=0,s=1){const e=(a-n)/(s-n);return W.y1-e*W.height}function Ms(a,n=0,s=1,e=0,t=1){if(a.length===0)return"";const i=[];let p=!1;for(const[o,c]of a){if(!Number.isFinite(c)){p=!1;continue}const u=us(o,n,s).toFixed(2),d=ms(c,e,t).toFixed(2);i.push(`${p?"L":"M"} ${u} ${d}`),p=!0}return i.join(" ")}function Da(a=1.5,n=64){const s=[];for(let e=0;e<n;e++){const t=e/(n-1);s.push([t,Math.sin(2*Math.PI*a*t)])}return s}function Ta(a=2){const n=[],s=1/a;for(let e=0;e<a;e++){const t=e*s;n.push([t,1]),n.push([t+s/2,1]),n.push([t+s/2,-1]),n.push([t+s,-1]),e<a-1&&n.push([t+s,1])}return n}function qa(a=2,n=81){const s=[];for(let e=0;e<n;e++){const t=e/(n-1),i=t*a*4%4,p=i<1?i:i<3?2-i:i-4;s.push([t,p])}return s}function Ca(a=.6,n=.5,s=.05,e=.15,t=.07){const i=[[0,0]];let p=s;for(;p<1;){i.push([p,0]),i.push([Math.min(1,p+e),1]);const o=p+a*n;i.push([Math.min(1,o),1]),i.push([Math.min(1,o+t),0]),p+=a}return i.push([1,0]),i}function Aa(a=.25){return[[0,0],[a,0],[a,1],[1,1]]}function Ma(a=.5,n=.13,s=80){const e=[];for(let t=0;t<s;t++){const i=t/(s-1);e.push([i,Math.exp(-((i-a)**2)/(2*n*n))])}return e}function Ia(a=1,n=6,s=120){const e=[];for(let t=0;t<s;t++){const i=t/(s-1),p=n-a,o=2*Math.PI*(a*i+.5*p*i*i);e.push([i,Math.sin(o)])}return e}function Pa(a=28,n=5){let s=n;const e=()=>(s=(s*9301+49297)%233280,s/233280),t=[];for(let p=0;p<a;p++){const o=Math.max(1e-6,e()),c=e();t.push(Math.sqrt(-2*Math.log(o))*Math.cos(2*Math.PI*c))}const i=Math.max(...t.map(Math.abs));return t.map((p,o)=>[o/(a-1),p/i*.95])}function Fa(a=35,n=11){let s=n;const e=()=>(s=(s*9301+49297)%233280,s/233280-.5),t=5,i=new Array(t).fill(0),p=new Array(t).fill(0),o=[];for(let u=0;u<a;u++){let d=0;for(let v=0;v<t;v++)p[v]===0&&(i[v]=e(),p[v]=1<<v),p[v]--,d+=i[v];o.push(d)}const c=Math.max(...o.map(Math.abs));return o.map((u,d)=>[d/(a-1),u/c*.9])}function La(a=1){return[[0,a],[1,a]]}function Oa(a=.32,n=.06){const s=[[0,0]];let e=n;for(;e<1;){s.push([e,0]),s.push([e,1]);const t=Math.min(1,e+a/2);s.push([t,1]),s.push([t,0]),e+=a}return s.push([1,0]),s}function za(a=.18,n=.15,s=60){const e=[];for(let t=0;t<s;t++){const i=t/(s-1);i<n?e.push([i,0]):e.push([i,1-Math.exp(-(i-n)/a)])}return e}function xs(a=.25,n=22,s=.15,e=100){const t=[],i=n*Math.sqrt(1-a*a),p=Math.atan2(Math.sqrt(1-a*a),a);for(let o=0;o<e;o++){const c=o/(e-1);if(c<s)t.push([c,0]);else{const u=c-s,d=Math.exp(-a*n*u)/Math.sqrt(1-a*a);t.push([c,1-d*Math.sin(i*u+p)])}}return t}function Na(a=60){const n=[];for(let o=0;o<a;o++){const c=o/(a-1);if(c<.15)n.push([c,0]);else{const u=c-.15,d=1.4*Math.exp(-u/.06)+1*(1-Math.exp(-u/.25));n.push([c,d])}}return n}function Ha(a=.15){return[[0,0],[a,0],[1,1-a]]}function Ea(a=1,n=64){const s=[];for(let e=0;e<n;e++){const t=e/(n-1);s.push([t,Math.sin(2*Math.PI*a*t)])}return s}function Ra(a=.25,n=1,s=56){const e=[[0,0],[a,0]];for(let t=1;t<s;t++){const i=a+(1-a)*(t/(s-1));e.push([i,Math.sin(2*Math.PI*n*(i-a))])}return e}function Ba(a=4,n=80){const s=[];for(let i=0;i<n;i++){const p=i/(n-1),o=Math.pow(10,-1.2+p*(1.2- -1.2));s.push([p,1/Math.sqrt(1+Math.pow(o,2*a))])}return s}function ja(a=4,n=80){const s=[];for(let i=0;i<n;i++){const p=i/(n-1),o=Math.pow(10,-1.2+p*(1.2- -1.2));s.push([p,Math.pow(o,a)/Math.sqrt(1+Math.pow(o,2*a))])}return s}function Wa(a=2,n=2,s=100){const e=[];for(let p=0;p<s;p++){const o=p/(s-1),c=Math.pow(10,-1.5+o*(1.5- -1.5)),u=Math.pow(c/n,a),d=Math.sqrt(Math.pow(1-c*c,2*a)+Math.pow(c/n,2*a));e.push([o,u/d])}return e}function Ka(a=2,n=2,s=100){const e=[];for(let p=0;p<s;p++){const o=p/(s-1),c=Math.pow(10,-1.5+o*(1.5- -1.5)),u=Math.pow(Math.abs(1-c*c),a),d=Math.sqrt(Math.pow(1-c*c,2*a)+Math.pow(c/n,2*a));e.push([o,u/d])}return e}function Ga(a=80){const n=[],t=Math.pow(10,1.2);for(let i=0;i<a;i++){const p=i/(a-1),o=Math.pow(10,-1.2+p*(1.2- -1.2));n.push([p,o/t])}return n}function Ua(a=80){const n=[];for(let s=0;s<a;s++){const e=s/(a-1),t=Math.pow(10,-1.2+e*2.4),i=Math.abs(Math.sin(t*1.2)/(t*1.2+1e-4)),p=1/Math.sqrt(1+Math.pow(t/1,6));n.push([e,i*p])}return n}function Za(a=80){const n=[];for(let s=0;s<a;s++){const e=-Math.PI+2*Math.PI*s/(a-1);n.push([e,Math.sin(e)])}return n}function Xa(a=80){const n=[];for(let s=0;s<a;s++){const e=-Math.PI+2*Math.PI*s/(a-1);n.push([e,Math.cos(e)])}return n}function Ya(a=4,n=80){const s=[];for(let e=0;e<n;e++){const t=-1+2*e/(n-1);s.push([t,Math.atan(t*a)/(Math.PI/2)])}return s}function Va(a=2,n=60){const s=[];for(let e=0;e<n;e++){const t=-1+2*e/(n-1);s.push([t,Math.pow(t,a)])}return s}function Is(a=6){const n=[];for(let s=0;s<a;s++){const e=s/a,t=(s+1)/a,i=s/(a-1);n.push([e,i]),n.push([t,i])}return n}function Ja(a=6){const n=[];for(let s=0;s<a;s++){const e=s/a,t=(s+1)/a,i=1-s/(a-1);n.push([e,i]),n.push([t,i])}return n}function Qa(a=.12,n=100){return xs(.4,18,a,n)}function $a(a=3){const n=[],s=1/a;for(let e=0;e<a;e++){const t=e*s;n.push([t,0]),n.push([t+s,1]),e<a-1&&n.push([t+s,0])}return n}function sn(a=1.55,n=240){const s=[];for(let e=0;e<n;e++){const t=-Math.PI+2*Math.PI*e/(n-1),i=Math.tan(t);!Number.isFinite(i)||Math.abs(i)>a?s.push([t,NaN]):s.push([t,i])}return s}function an(){return[[-1,-.6],[-.5,-.55],[-.05,0],[.4,.55],[.75,.75],[1,.78]]}function nn(a=.4,n=80){const s=[];for(let i=0;i<n;i++){const p=i/(n-1),o=Math.pow(10,-1.2+p*(1.2- -1.2)),c=Math.sqrt(Math.pow(1-o*o,2)+Math.pow(2*a*o,2));s.push([p,1/c])}return s}function en(a=4,n=60){const s=[];for(let e=0;e<n;e++){const t=-1+2*(e/(n-1));s.push([t,Math.tanh(t*a)])}return s}function tn(a=60){const n=[];for(let s=0;s<a;s++){const e=s/(a-1);n.push([e,(Math.exp(e*2.2)-1)/(Math.exp(2.2)-1)])}return n}function Ps(a=60){const n=[];for(let s=0;s<a;s++){const e=s/(a-1);n.push([e,Math.log(1+9*e)/Math.log(10)])}return n}function ln(a=60){const n=[];for(let s=0;s<a;s++){const e=s/(a-1);n.push([e,Math.sqrt(e)])}return n}function pn(a=60){const n=[];for(let s=0;s<a;s++){const e=-1+2*(s/(a-1));n.push([e,Math.abs(e)])}return n}function on(a=.6){return[[-1,-a],[-a,-a],[a,a],[1,a]]}function rn(a=.3){return[[-1,-1+a],[-a,0],[a,0],[1,1-a]]}function cn(a=.3){return[[-1,-1],[a,-1],[a,1],[1,1],[-a,1],[-a,-1],[-1,-1]]}function dn(){return[[0,0],[.15,0],[.15+.45,1],[1,1]]}function un(a=6){const n=[],s=[.1,.3,.55,.75,.55,.85];for(let e=0;e<a;e++){const t=e/a,i=(e+1)/a;n.push([t,s[e]]),n.push([i,s[e]])}return n}function mn(a=6){const n=[.1,.3,.55,.75,.55,.85],s=[];for(let e=0;e<a;e++)s.push([e/(a-1),n[e]]);return s}function fn(a=.42,n=.8){return[[-1,-n],[-1+2*a,-n],[1,n],[1-2*a,n],[-1,-n]]}function hn(a=11,n=7){let s=n;const e=()=>(s=(s*9301+49297)%233280,s/233280),t=[];for(let i=0;i<a;i++)t.push([(i+.5)/a,e()*1.8-.9]);return t}const gn=.85,bn=1.15;function Ss(a){return gn*Math.sin(2*Math.PI*bn*a)}function Fs(a=90){return Array.from({length:a},(n,s)=>{const e=s/(a-1);return[e,Ss(e)]})}function vn(a=7){return Array.from({length:a},(n,s)=>{const e=(s+.5)/a;return[e,Ss(e)]})}function yn(a=7){const n=[];for(let s=0;s<a;s++){const e=Ss((s+.5)/a);n.push([s/a,e]),n.push([(s+1)/a,e])}return n}function kn(a=1.45,n=3,s=200){const e=[];for(let t=0;t<s;t++){const i=t/(s-1);e.push([i,Math.exp(-a*i)*Math.sin(2*Math.PI*n*i)])}return e}function wn(){return[[.1,.28],[.22,1],[.34,.2],[.46,.62],[.58,.16],[.7,.4],[.82,.14],[.94,.24]]}function _n(a=.82,n=.12,s=100){return xs(.4,18,n,s).map(([e,t])=>[e,Math.min(t,a)])}const ss=[-1.05,1.05],j=[-1.1,1.1],Ls=[-.7,.7],ws=[0,1.5],xn=[0,1.6],_s=[-Math.PI*1.05,Math.PI*1.05],Sn=[0,1.6],Gs={Constant:{kind:"plot",samples:()=>La()},StepSource:{kind:"plot",samples:()=>Aa()},SinusoidalSource:{kind:"plot",samples:()=>Da(),yRange:j},SquareWaveSource:{kind:"plot",samples:()=>Ta(),yRange:j},TriangleWaveSource:{kind:"plot",samples:()=>qa(),yRange:j},PulseSource:{kind:"plot",samples:()=>Ca()},GaussianPulseSource:{kind:"plot",samples:()=>Ma()},ChirpPhaseNoiseSource:{kind:"plot",samples:()=>Ia(),yRange:j},WhiteNoise:{kind:"plot",samples:()=>Pa(),yRange:j},PinkNoise:{kind:"plot",samples:()=>Fa(),yRange:j},RandomNumberGenerator:{kind:"plot",samples:()=>hn(),yRange:j,stems:!0},ClockSource:{kind:"plot",samples:()=>Oa()},Source:{kind:"math",latex:"f(t)"},PT1:{kind:"plot",samples:()=>za()},PT2:{kind:"plot",samples:()=>xs(),yRange:ws},LeadLag:{kind:"plot",samples:()=>Na(),yRange:xn},Integrator:{kind:"plot",samples:()=>Ha()},Differentiator:{kind:"plot",samples:()=>Ga()},Delay:{kind:"plot",samples:()=>Ra(),samplesDashed:()=>Ea(),yRange:j},PID:{kind:"plot",samples:()=>Qa(),yRange:ws},AntiWindupPID:{kind:"plot",samples:()=>_n(),yRange:ws},ButterworthLowpassFilter:{kind:"plot",samples:()=>Ba()},ButterworthHighpassFilter:{kind:"plot",samples:()=>ja()},ButterworthBandpassFilter:{kind:"plot",samples:()=>Wa()},ButterworthBandstopFilter:{kind:"plot",samples:()=>Ka()},FIR:{kind:"plot",samples:()=>Ua()},TransferFunctionNumDen:{kind:"plot",samples:()=>nn(.35),yRange:Sn},TransferFunctionZPG:{kind:"pz"},Tanh:{kind:"plot",samples:()=>en(),xRange:ss,yRange:j},Exp:{kind:"plot",samples:()=>tn()},Log:{kind:"plot",samples:()=>Ps()},Log10:{kind:"plot",samples:()=>Ps(),badge:"10"},Sqrt:{kind:"plot",samples:()=>ln()},Abs:{kind:"plot",samples:()=>pn(),xRange:ss,axes:"baseline"},Clip:{kind:"plot",samples:()=>on(),xRange:ss,yRange:Ls},Deadband:{kind:"plot",samples:()=>rn(),xRange:ss,yRange:Ls,axes:"yaxis"},Relay:{kind:"plot",samples:()=>cn(.45),xRange:ss,yRange:j,axes:"none"},RateLimiter:{kind:"plot",samples:()=>dn()},SampleHold:{kind:"plot",samples:()=>un()},Backlash:{kind:"plot",samples:()=>fn(),xRange:ss,yRange:j},Sin:{kind:"plot",samples:()=>Za(),xRange:_s,yRange:j},Cos:{kind:"plot",samples:()=>Xa(),xRange:_s,yRange:j},Tan:{kind:"plot",samples:()=>sn(),xRange:_s,yRange:[-1.6,1.6],asymptotes:[-Math.PI/2,Math.PI/2]},Pow:{kind:"plot",samples:()=>Va(2),xRange:ss},Mod:{kind:"plot",samples:()=>$a()},Atan2:{kind:"plot",samples:()=>Ya(),xRange:ss,yRange:[-1.25,1.25]},ADC:{kind:"plot",samples:()=>vn(),samplesDashed:()=>Fs(),yRange:j,stems:!0},DAC:{kind:"plot",samples:()=>yn(),samplesDashed:()=>Fs(),yRange:j},Counter:{kind:"plot",samples:()=>Is()},CounterUp:{kind:"plot",samples:()=>Is(),decoration:"arrow-up"},CounterDown:{kind:"plot",samples:()=>Ja(),decoration:"arrow-down"},LUT1D:{kind:"plot",samples:()=>an(),xRange:ss,yRange:j,markers:!0},LUT:{kind:"surface",fn:(a,n)=>-.18*(a+n)+.3*a*n},ODE:{kind:"math",latex:"\\dot{x} = f(x, u, t)"},StateSpace:{kind:"math",latex:"\\dot{x} = Ax{+}Bu"},DynamicalSystem:{kind:"math",latex:"\\begin{aligned}\\dot{x} &= f\\\\ y &= g\\end{aligned}"},DynamicalFunction:{kind:"math",latex:"f(u, t)"},Function:{kind:"math",latex:"f(u)"},Polynomial:{kind:"math",latex:"\\sum c_k\\,u^{k}"},FirstOrderHold:{kind:"plot",samples:()=>mn(),markers:!0},DiscreteIntegrator:{kind:"math",latex:"\\dfrac{T}{z-1}"},DiscreteDerivative:{kind:"math",latex:"\\dfrac{z-1}{T\\,z}"},DiscreteStateSpace:{kind:"math",latex:"x_{k+1} = Ax_k{+}Bu_k"},DiscreteTransferFunction:{kind:"math",latex:"\\dfrac{B(z)}{A(z)}"},TappedDelay:{kind:"svg",name:"TappedDelay"},Adder:{kind:"svg",name:"Adder"},Multiplier:{kind:"svg",name:"Multiplier"},Amplifier:{kind:"svg",name:"Amplifier"},Rescale:{kind:"svg",name:"Amplifier"},Divider:{kind:"svg",name:"Divider"},LogicAnd:{kind:"svg",name:"LogicAnd"},LogicOr:{kind:"svg",name:"LogicOr"},LogicNot:{kind:"svg",name:"LogicNot"},Equal:{kind:"svg",name:"Equal"},GreaterThan:{kind:"svg",name:"GreaterThan"},LessThan:{kind:"svg",name:"LessThan"},Alias:{kind:"svg",name:"Alias"},Wrapper:{kind:"svg",name:"Wrapper"},Switch:{kind:"svg",name:"Switch"},Subsystem:{kind:"svg",name:"Subsystem"},Interface:{kind:"svg",name:"Interface"},Scope:{kind:"scope",samples:()=>kn(),yRange:[-1.15,1.15],gridX:4,gridY:2},Spectrum:{kind:"scope",samples:()=>wn(),yRange:[0,1.12],gridX:0,gridY:2,bars:!0}};function Dn(a){return a?Gs[a]:void 0}function Tn(a){return!!a&&a in Gs}var qn=I("<line></line>"),Cn=I("<line></line>"),An=I('<g class="axis svelte-1384908"><!><!></g>'),Mn=I('<line class="asymptote svelte-1384908"></line>'),In=I('<path class="ghost svelte-1384908" stroke-dasharray="3.5 3"></path>'),Pn=I('<line></line><circle r="2.4" fill="currentColor" stroke="none"></circle>',1),Fn=I("<path></path>"),Ln=I('<circle r="2.8" fill="currentColor" stroke="none"></circle>'),On=I('<path d="M 86 44 L 86 22 M 82 26 L 86 22 L 90 26"></path>'),zn=I('<path d="M 86 22 L 86 44 M 82 40 L 86 44 L 90 40"></path>'),Nn=I(`<text text-anchor="start" dominant-baseline="hanging" fill="currentColor" stroke="none" font-family="ui-monospace, 'JetBrains Mono', 'SF Mono', Menlo, monospace" font-size="11" font-weight="600"> </text>`),Hn=I('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" class="svelte-1384908"><!><!><!><!><!><!><!></svg>');function En(a,n){os(n,!0);let s=K(n,"xRange",19,()=>[0,1]),e=K(n,"yRange",19,()=>[0,1]),t=K(n,"axes",3,"cross"),i=K(n,"markers",3,!1),p=K(n,"stems",3,!1);const o=q(()=>Ms(n.samples,s()[0],s()[1],e()[0],e()[1])),c=q(()=>n.samplesDashed?Ms(n.samplesDashed,s()[0],s()[1],e()[0],e()[1]):""),u=q(()=>e()[0]<=0&&e()[1]>=0?ms(0,e()[0],e()[1]):Y.y1),d=q(()=>s()[0]<0&&s()[1]>0?us(0,s()[0],s()[1]):Y.x0),v=q(()=>n.samples.filter(([,h])=>Number.isFinite(h))),A=q(()=>(n.asymptotes??[]).map(h=>us(h,s()[0],s()[1])));var P=Hn(),O=U(P);{var S=h=>{var x=An(),b=U(x);{var _=R=>{var M=qn();L(()=>{r(M,"x1",Y.x0),r(M,"y1",l(u)),r(M,"x2",Y.x1),r(M,"y2",l(u))}),g(R,M)};H(b,R=>{(t()==="baseline"||t()==="cross")&&R(_)})}var D=B(b);{var z=R=>{var M=Cn();L(()=>{r(M,"x1",l(d)),r(M,"y1",Y.y0),r(M,"x2",l(d)),r(M,"y2",Y.y1)}),g(R,M)};H(D,R=>{(t()==="yaxis"||t()==="cross")&&R(z)})}Z(x),g(h,x)};H(O,h=>{t()!=="none"&&h(S)})}var C=B(O);as(C,17,()=>l(A),ns,(h,x)=>{var b=Mn();L(()=>{r(b,"x1",l(x)),r(b,"y1",W.y0),r(b,"x2",l(x)),r(b,"y2",W.y1)}),g(h,b)});var w=B(C);{var f=h=>{var x=In();L(()=>r(x,"d",l(c))),g(h,x)};H(w,h=>{l(c)&&h(f)})}var y=B(w);{var k=h=>{var x=X(),b=G(x);as(b,17,()=>l(v),ns,(_,D)=>{var z=q(()=>fs(l(D),2));let R=()=>l(z)[0],M=()=>l(z)[1];const N=q(()=>us(R(),s()[0],s()[1]));var Q=Pn(),V=G(Q),$=B(V);L((ts,ps)=>{r(V,"x1",l(N)),r(V,"y1",l(u)),r(V,"x2",l(N)),r(V,"y2",ts),r($,"cx",l(N)),r($,"cy",ps)},[()=>ms(M(),e()[0],e()[1]),()=>ms(M(),e()[0],e()[1])]),g(_,Q)}),g(h,x)},m=h=>{var x=Fn();L(()=>r(x,"d",l(o))),g(h,x)};H(y,h=>{p()?h(k):h(m,!1)})}var T=B(y);{var F=h=>{var x=X(),b=G(x);as(b,17,()=>l(v),ns,(_,D)=>{var z=q(()=>fs(l(D),2));let R=()=>l(z)[0],M=()=>l(z)[1];var N=Ln();L((Q,V)=>{r(N,"cx",Q),r(N,"cy",V)},[()=>us(R(),s()[0],s()[1]),()=>ms(M(),e()[0],e()[1])]),g(_,N)}),g(h,x)};H(T,h=>{i()&&h(F)})}var E=B(T);{var es=h=>{var x=On();g(h,x)},is=h=>{var x=X(),b=G(x);{var _=D=>{var z=zn();g(D,z)};H(b,D=>{n.decoration==="arrow-down"&&D(_)},!0)}g(h,x)};H(E,h=>{n.decoration==="arrow-up"?h(es):h(is,!1)})}var J=B(E);{var ls=h=>{var x=Nn(),b=U(x,!0);Z(x),L(()=>{r(x,"x",Y.x0+4),r(x,"y",Y.y0),Rs(b,n.badge)}),g(h,x)};H(J,h=>{n.badge&&h(ls)})}Z(P),g(a,P),rs()}let bs=null;async function Rn(){return bs||(bs=await Qs(()=>import("./XbL3y5x-.js"),[],import.meta.url),bs)}function ye(){return"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"}var Bn=Bs('<span class="math svelte-1io0tjp"><span class="inner svelte-1io0tjp"><!></span></span>');function jn(a,n){os(n,!0);let s=hs(""),e=hs(void 0),t=hs(void 0),i=hs(1);const p=1.6,o=.4,c=6,u=4;async function d(){if(await Hs(),!l(e)||!l(t))return;const S=l(t).clientWidth-2*c,C=l(t).clientHeight-2*u,w=l(e).scrollWidth,f=l(e).scrollHeight;if(w===0||f===0||S<=0||C<=0)return;const y=Math.min(S/w,C/f);ds(i,Math.max(o,Math.min(p,y)),!0)}Ys(async()=>{const S=await Rn();try{ds(s,S.default.renderToString(n.latex,{displayMode:!0,throwOnError:!1,strict:!1,output:"html"}),!0)}catch{ds(s,n.latex,!0)}await d()}),Ds(()=>{l(s)&&d()}),Ds(()=>{if(!l(t))return;const S=new ResizeObserver(()=>d());return S.observe(l(t)),()=>S.disconnect()});var v=Bn(),A=U(v),P=U(A);{var O=S=>{var C=X(),w=G(C);js(w,()=>l(s)),g(S,C)};H(P,S=>{l(s)&&S(O)})}Z(A),Ts(A,S=>ds(e,S),()=>l(e)),Z(v),Ts(v,S=>ds(t,S),()=>l(t)),L(()=>Js(A,`transform: scale(${l(i)??""});`)),g(a,v),rs()}var Wn=I(`<svg xmlns="http://www.w3.org/2000/svg" class="svelte-61qc8h"><text text-anchor="middle" dominant-baseline="central" fill="currentColor" stroke="none" font-family="ui-monospace, 'JetBrains Mono', 'SF Mono', Menlo, monospace" letter-spacing="-1"> </text></svg>`);function Kn(a,n){let s=K(n,"size",3,.45),e=K(n,"bold",3,!0);const t=96,i=64,p=q(()=>i*s());var o=Wn();r(o,"viewBox","0 0 96 64");var c=U(o);r(c,"x",t/2),r(c,"y",i/2);var u=U(c,!0);Z(c),Z(o),L(()=>{r(c,"font-size",l(p)),r(c,"font-weight",e()?700:500),Rs(u,n.text)}),g(a,o)}var Gn=I("<line></line>"),Un=I("<line></line>"),Zn=I('<line class="bar svelte-odzmgb"></line>'),Xn=I('<!><line class="baseline svelte-odzmgb"></line>',1),Yn=I('<path stroke-width="1.6" stroke-dasharray="4 4" stroke-dashoffset="2"></path>'),Vn=I('<path stroke-width="1.6"></path><!>',1),Jn=I('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" class="svelte-odzmgb"><rect rx="4" stroke-width="1.6"></rect><g class="grid svelte-odzmgb"><!><!></g><!></svg>');function Qn(a,n){os(n,!0);let s=K(n,"yRange",19,()=>[-1.1,1.1]),e=K(n,"gridX",3,4),t=K(n,"gridY",3,3),i=K(n,"bars",3,!1);const p={x0:6,x1:90,y0:7,y1:57},o=p.x1-p.x0,c=p.y1-p.y0,u=7,d=6,v=p.x0+u,A=p.x1-u,P=p.y0+d,O=p.y1-d;function S(b){return v+b*(A-v)}function C(b){const _=(b-s()[0])/(s()[1]-s()[0]);return O-_*(O-P)}function w(b){return b.map(([_,D],z)=>`${z===0?"M":"L"} ${S(_).toFixed(2)} ${C(D).toFixed(2)}`).join(" ")}const f=q(()=>i()?"":w(n.samples)),y=q(()=>n.samples2?w(n.samples2):""),k=q(()=>e()>1?Array.from({length:e()-1},(b,_)=>p.x0+(_+1)*o/e()):[]),m=q(()=>t()>1?Array.from({length:t()-1},(b,_)=>p.y0+(_+1)*c/t()):[]),T=q(()=>C(s()[0]));var F=Jn(),E=U(F),es=B(E),is=U(es);as(is,17,()=>l(k),ns,(b,_)=>{var D=Gn();L(()=>{r(D,"x1",l(_)),r(D,"y1",p.y0+2),r(D,"x2",l(_)),r(D,"y2",p.y1-2)}),g(b,D)});var J=B(is);as(J,17,()=>l(m),ns,(b,_)=>{var D=Un();L(()=>{r(D,"x1",p.x0+2),r(D,"y1",l(_)),r(D,"x2",p.x1-2),r(D,"y2",l(_))}),g(b,D)}),Z(es);var ls=B(es);{var h=b=>{var _=Xn(),D=G(_);as(D,17,()=>n.samples,ns,(R,M)=>{var N=q(()=>fs(l(M),2));let Q=()=>l(N)[0],V=()=>l(N)[1];var $=Zn();L((ts,ps,vs)=>{r($,"x1",ts),r($,"y1",l(T)),r($,"x2",ps),r($,"y2",vs)},[()=>S(Q()),()=>S(Q()),()=>C(V())]),g(R,$)});var z=B(D);L(()=>{r(z,"x1",v),r(z,"y1",l(T)),r(z,"x2",A),r(z,"y2",l(T))}),g(b,_)},x=b=>{var _=Vn(),D=G(_),z=B(D);{var R=M=>{var N=Yn();L(()=>r(N,"d",l(y))),g(M,N)};H(z,M=>{l(y)&&M(R)})}L(()=>r(D,"d",l(f))),g(b,_)};H(ls,b=>{i()?b(h):b(x,!1)})}Z(F),L(()=>{r(E,"x",p.x0),r(E,"y",p.y0),r(E,"width",o),r(E,"height",c)}),g(a,F),rs()}var $n=I('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" class="svelte-16dl6zq"><path stroke-width="1.5"></path><path stroke-width="1.5"></path></svg>');function se(a,n){os(n,!0);let s=K(n,"rows",3,5),e=K(n,"cols",3,5),t=K(n,"fn",3,(f,y)=>.5*(f*f-y*y));const i=48,p=36,o=17,c=.45,u=11;function d(f,y,k){const m=i+(f-y)*o,T=p+(f+y)*o*c-k*u;return[m,T]}const v=q(()=>{const f=[];for(let y=0;y<s();y++){const k=-1+2*y/(s()-1),m=[];for(let T=0;T<e();T++){const F=-1+2*T/(e()-1);m.push(d(F,k,t()(F,k)))}f.push(m)}return f});function A(f){const y=[];for(let k=0;k<f.length;k++){y.push(`M ${f[k][0][0].toFixed(2)} ${f[k][0][1].toFixed(2)}`);for(let m=1;m<f[k].length;m++)y.push(`L ${f[k][m][0].toFixed(2)} ${f[k][m][1].toFixed(2)}`)}for(let k=0;k<f[0].length;k++){y.push(`M ${f[0][k][0].toFixed(2)} ${f[0][k][1].toFixed(2)}`);for(let m=1;m<f.length;m++)y.push(`L ${f[m][k][0].toFixed(2)} ${f[m][k][1].toFixed(2)}`)}return y.join(" ")}const P=q(()=>A(l(v))),O=q(()=>{if(l(v).length===0)return"";const f=s()-1,y=e()-1,k=[];for(let m=0;m<=y;m++)k.push(l(v)[0][m]);for(let m=1;m<=f;m++)k.push(l(v)[m][y]);for(let m=y-1;m>=0;m--)k.push(l(v)[f][m]);for(let m=f-1;m>=1;m--)k.push(l(v)[m][0]);return k.map(([m,T],F)=>`${F===0?"M":"L"} ${m.toFixed(2)} ${T.toFixed(2)}`).join(" ")+" Z"});var S=$n(),C=U(S),w=B(C);Z(S),L(()=>{r(C,"d",l(P)),r(w,"d",l(O))}),g(a,S),rs()}var ae=I("<path></path>"),ne=I("<circle></circle>"),ee=I('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" class="svelte-9ooiz"><g class="axis svelte-9ooiz"><line></line><line></line></g><!><!></svg>');function te(a,n){os(n,!0);let s=K(n,"poles",19,()=>[[-.55,.6],[-.55,-.6]]),e=K(n,"zeros",19,()=>[[.45,0]]);const t=(W.x0+W.x1)/2,i=(W.y0+W.y1)/2,p=W.width/2,o=W.height/2,c=w=>t+w*p,u=w=>i-w*o,d=3.4;var v=ee(),A=U(v),P=U(A),O=B(P);Z(A);var S=B(A);as(S,17,s,ns,(w,f)=>{var y=q(()=>fs(l(f),2));let k=()=>l(y)[0],m=()=>l(y)[1];const T=q(()=>c(k())),F=q(()=>u(m()));var E=ae();L(()=>r(E,"d",`M ${l(T)-d} ${l(F)-d} L ${l(T)+d} ${l(F)+d} M ${l(T)+d} ${l(F)-d} L ${l(T)-d} ${l(F)+d}`)),g(w,E)});var C=B(S);as(C,17,e,ns,(w,f)=>{var y=q(()=>fs(l(f),2));let k=()=>l(y)[0],m=()=>l(y)[1];var T=ne();r(T,"r",d),L((F,E)=>{r(T,"cx",F),r(T,"cy",E)},[()=>c(k()),()=>u(m())]),g(w,T)}),Z(v),L(()=>{r(P,"x1",Y.x0),r(P,"y1",i),r(P,"x2",Y.x1),r(P,"y2",i),r(O,"x1",t),r(O,"y1",Y.y0),r(O,"x2",t),r(O,"y2",Y.y1)}),g(a,v),rs()}const ie=Object.assign({"./blocks/svg/Adder.svg":ra,"./blocks/svg/Alias.svg":ca,"./blocks/svg/Amplifier.svg":da,"./blocks/svg/Divider.svg":ua,"./blocks/svg/Equal.svg":ma,"./blocks/svg/GreaterThan.svg":fa,"./blocks/svg/Interface.svg":ha,"./blocks/svg/LessThan.svg":ga,"./blocks/svg/LogicAnd.svg":ba,"./blocks/svg/LogicNot.svg":va,"./blocks/svg/LogicOr.svg":ya,"./blocks/svg/Multiplier.svg":ka,"./blocks/svg/Subsystem.svg":wa,"./blocks/svg/Switch.svg":_a,"./blocks/svg/TappedDelay.svg":xa,"./blocks/svg/Wrapper.svg":Sa}),Us=new Map;for(const[a,n]of Object.entries(ie)){const s=a.match(/\/([^/]+)\.svg$/);s&&Us.set(s[1],n)}function ke(a){return Tn(a)}var le=Bs('<span class="block-icon svelte-kd944p"><!></span>');function we(a,n){os(n,!0);const s=q(()=>Dn(n.blockClass)),e=q(()=>l(s)?.kind==="svg"?Us.get(l(s).name):void 0);var t=X(),i=G(t);{var p=o=>{var c=le(),u=U(c);{var d=A=>{{let P=q(()=>l(s).samples()),O=q(()=>l(s).samplesDashed?.());En(A,{get samples(){return l(P)},get samplesDashed(){return l(O)},get xRange(){return l(s).xRange},get yRange(){return l(s).yRange},get axes(){return l(s).axes},get markers(){return l(s).markers},get decoration(){return l(s).decoration},get asymptotes(){return l(s).asymptotes},get badge(){return l(s).badge},get stems(){return l(s).stems}})}},v=A=>{var P=X(),O=G(P);{var S=w=>{te(w,{get poles(){return l(s).poles},get zeros(){return l(s).zeros}})},C=w=>{var f=X(),y=G(f);{var k=T=>{{let F=q(()=>l(s).samples()),E=q(()=>l(s).samples2?.());Qn(T,{get samples(){return l(F)},get samples2(){return l(E)},get yRange(){return l(s).yRange},get gridX(){return l(s).gridX},get gridY(){return l(s).gridY},get bars(){return l(s).bars}})}},m=T=>{var F=X(),E=G(F);{var es=J=>{se(J,{get fn(){return l(s).fn},get rows(){return l(s).rows},get cols(){return l(s).cols}})},is=J=>{var ls=X(),h=G(ls);{var x=_=>{jn(_,{get latex(){return l(s).latex}})},b=_=>{var D=X(),z=G(D);{var R=N=>{Kn(N,{get text(){return l(s).text},get size(){return l(s).size}})},M=N=>{var Q=X(),V=G(Q);{var $=ts=>{var ps=X(),vs=G(ps);js(vs,()=>l(e)),g(ts,ps)};H(V,ts=>{l(s).kind==="svg"&&l(e)&&ts($)},!0)}g(N,Q)};H(z,N=>{l(s).kind==="glyph"?N(R):N(M,!1)},!0)}g(_,D)};H(h,_=>{l(s).kind==="math"?_(x):_(b,!1)},!0)}g(J,ls)};H(E,J=>{l(s).kind==="surface"?J(es):J(is,!1)},!0)}g(T,F)};H(y,T=>{l(s).kind==="scope"?T(k):T(m,!1)},!0)}g(w,f)};H(O,w=>{l(s).kind==="pz"?w(S):w(C,!1)},!0)}g(A,P)};H(u,A=>{l(s).kind==="plot"?A(d):A(v,!1)})}Z(c),L(()=>{r(c,"aria-label",n.title),r(c,"role",n.title?"img":void 0)}),g(o,c)};H(i,o=>{l(s)&&o(p)})}g(a,t),rs()}export{we as B,$s as D,qs as P,Dn as a,fe as b,be as c,sa as d,aa as e,he as f,ye as g,ke as h,Ws as i,ge as j,na as k,Rn as l,me as m,la as n,ve as r};
