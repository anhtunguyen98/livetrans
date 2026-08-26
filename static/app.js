const $ = (selector) => document.querySelector(selector);
const state = { recorder: null, stream: null, chunks: [], startedAt: 0, timer: null, audioUrl: null, audioUrls: [], audioContext: null, analyser: null, raf: null };
const timeline = { input: [], output: [] };
const els = {
  record: $("#recordButton"), file: $("#audioFile"), canvas: $("#waveform"),
  transcript: $("#transcript"), translation: $("#translation"), play: $("#playButton"),
  timer: $("#timer"), stage: $("#stageLabel"), source: $("#sourceLanguage"), target: $("#targetLanguage"),
  speed: $("#speed"), speedValue: $("#speedValue"), toast: $("#toast")
};

async function init() {
  drawTimeline();
  try {
    const [status, health] = await Promise.all([
      fetch("/api/status").then(r => r.json()),
      fetch("/api/vllm/health").then(r => r.json()),
    ]);
    const servicesReady = health.asr && health.translation && health.speech;
    $("#runtimeText").textContent = status.mode === "mock"
      ? "Mock mode · UI preview"
      : servicesReady ? "ASR + MT + TTS connected" : `Offline · ASR ${health.asr ? "✓" : "×"} · MT ${health.translation ? "✓" : "×"} · TTS ${health.speech ? "✓" : "×"}`;
    $(".runtime").classList.toggle("ready", status.mode === "mock" || servicesReady);
    $("#asrModel").textContent = shortName(status.models.asr); $("#mtModel").textContent = shortName(status.models.translation); $("#ttsModel").textContent = shortName(status.models.speech);
  } catch { showToast("Không kết nối được backend."); }
}

function shortName(name) { return name.split("/").pop().replace("-hf", ""); }

els.record.addEventListener("click", () => state.recorder?.state === "recording" ? stopRecording() : startRecording());
els.file.addEventListener("change", () => { const file = els.file.files[0]; if (file) processAudio(file); });
els.play.addEventListener("click", () => playUrls(state.audioUrls.length ? state.audioUrls : [state.audioUrl].filter(Boolean)));
els.speed.addEventListener("input", () => els.speedValue.value = `${Number(els.speed.value).toFixed(2)}×`);
$("#swapLanguages").addEventListener("click", () => { if (els.source.value === "auto") { showToast("Chọn ngôn ngữ nguồn trước khi đổi."); return; } [els.source.value, els.target.value] = [els.target.value, els.source.value]; });

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) return showToast("Trình duyệt không hỗ trợ microphone.");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    state.chunks = []; state.recorder = new MediaRecorder(state.stream);
    state.recorder.ondataavailable = e => e.data.size && state.chunks.push(e.data);
    state.recorder.onstop = () => processAudio(new Blob(state.chunks, { type: state.recorder.mimeType }), "recording.webm");
    state.recorder.start(250); state.startedAt = Date.now();
    els.record.classList.add("recording"); els.record.querySelector("b").textContent = "Dừng và dịch"; els.stage.textContent = "Đang nghe";
    state.timer = setInterval(updateTimer, 250); connectWaveform(state.stream);
  } catch (error) { showToast(`Không mở được microphone: ${error.message}`); }
}

function stopRecording() {
  state.recorder.stop(); state.stream.getTracks().forEach(track => track.stop()); clearInterval(state.timer); cancelAnimationFrame(state.raf);
  els.record.classList.remove("recording"); els.record.querySelector("b").textContent = "Bắt đầu nói";
}

async function processAudio(blob, name = blob.name || "audio.webm") {
  setProcessing(true); $("#textTTFT").textContent="TTFT —"; $("#audioTTFA").textContent="TTFA —"; const form = new FormData();
  form.append("audio", blob, name); form.append("source_language", els.source.value); form.append("target_language", els.target.value);
  form.append("speed", els.speed.value);
  try {
    const response = await fetch("/api/translate", { method: "POST", body: form }); const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Không dịch được audio.");
    setCopy(els.transcript, data.transcript); setCopy(els.translation, data.translation); $("#detectedLanguage").textContent = data.detected_language; $("#latency").textContent = `${data.latency_ms} ms`;
    const binary = atob(data.audio); const bytes = Uint8Array.from(binary, c => c.charCodeAt(0)); state.audioUrls.forEach(URL.revokeObjectURL); state.audioUrl = URL.createObjectURL(new Blob([bytes], { type: data.audio_mime })); state.audioUrls=[state.audioUrl]; els.play.disabled = false; $("#outputPreview").src=state.audioUrl;
    updateFeed(data.transcript,data.translation,true);
    $("#textTTFT").textContent=`TTFT ${data.time_to_first_text_ms} ms`; $("#audioTTFA").textContent=`TTFA ${data.time_to_first_audio_ms} ms`;
    els.stage.textContent = data.mode === "mock" ? "Hoàn tất · mock output" : "Hoàn tất"; drawResultWave();
  } catch (error) { showToast(error.message); els.stage.textContent = "Có lỗi · thử lại"; drawIdleWave(); }
  finally { setProcessing(false); }
}

async function playUrls(urls) { for (const url of urls) await new Promise(resolve => { const audio=new Audio(url); audio.onended=resolve; audio.onerror=resolve; audio.play().catch(resolve); }); }

function updateFeed(source, translation, final=false) { const feed=$("#translationFeed"); let card=feed.querySelector(".draft"); if(!card){feed.querySelector(":scope > .empty")?.remove();card=document.createElement("article");feed.appendChild(card);} card.className=`feed-card ${final?"":"draft"}`;card.innerHTML=`<p class="src"></p><p class="dst"></p>`;card.querySelector(".src").textContent=source;card.querySelector(".dst").textContent=translation; if(final)card.removeAttribute("data-draft"); feed.scrollTop=feed.scrollHeight; }

function setCopy(element, value) { element.textContent = value; element.classList.remove("empty"); }
function setProcessing(on) { els.record.classList.toggle("processing", on); els.record.disabled = on; els.stage.textContent = on ? "ASR → Translate → Speech" : els.stage.textContent; }
function updateTimer() { const seconds = Math.floor((Date.now() - state.startedAt)/1000); els.timer.textContent = `${String(Math.floor(seconds/60)).padStart(2,"0")}:${String(seconds%60).padStart(2,"0")}`; }
function connectWaveform(stream) { state.audioContext = new AudioContext(); state.analyser = state.audioContext.createAnalyser(); state.analyser.fftSize = 256; state.audioContext.createMediaStreamSource(stream).connect(state.analyser); drawLiveWave(); }
function canvasSetup() { const rect = els.canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1; els.canvas.width = rect.width*dpr; els.canvas.height = rect.height*dpr; const ctx=els.canvas.getContext("2d"); ctx.scale(dpr,dpr); return [ctx,rect.width,rect.height]; }
function drawLiveWave() { const [ctx,w,h]=canvasSetup(), data=new Uint8Array(state.analyser.frequencyBinCount); state.analyser.getByteFrequencyData(data); ctx.clearRect(0,0,w,h); ctx.fillStyle="#26c6c8"; const bar=w/data.length; data.forEach((v,i)=>{const bh=Math.max(2,v/255*h*.8);ctx.fillRect(i*bar,h/2-bh/2,Math.max(1,bar-2),bh)}); state.raf=requestAnimationFrame(drawLiveWave); }
function drawIdleWave() { const [ctx,w,h]=canvasSetup(); ctx.clearRect(0,0,w,h); ctx.strokeStyle="rgba(38,198,200,.5)";ctx.lineWidth=2;ctx.beginPath();for(let x=0;x<w;x+=3){const y=h/2+Math.sin(x*.035)*5+Math.sin(x*.11)*2; x?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke(); }
function drawResultWave() { const [ctx,w,h]=canvasSetup();ctx.clearRect(0,0,w,h);ctx.strokeStyle="#26c6c8";ctx.lineWidth=2;ctx.beginPath();for(let x=0;x<w;x+=2){const amp=Math.sin(x*.012)*28+42;const y=h/2+Math.sin(x*.13)*amp*Math.sin(Math.PI*x/w);x?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke(); }
function recordTimeline(kind,value){const list=timeline[kind];if(!list)return;list.push({time:performance.now(),value:Math.max(0,Math.min(1,value))});}
function drawTimeline(){const [ctx,w,h]=canvasSetup(),now=performance.now(),span=16000;ctx.clearRect(0,0,w,h);ctx.font='11px SFMono-Regular, monospace';[['input','#2fd3ff',h*.27],['output','#7b7dff',h*.73]].forEach(([kind,color,mid])=>{ctx.fillStyle='rgba(255,255,255,.025)';ctx.fillRect(0,mid-h*.2,w,h*.4);ctx.strokeStyle='rgba(255,255,255,.08)';ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(w,mid);ctx.stroke();ctx.fillStyle='rgba(135,161,193,.8)';ctx.fillText(kind.toUpperCase(),12,mid-h*.13);const list=timeline[kind];while(list.length&&now-list[0].time>span)list.shift();if(!list.length)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();list.forEach((p,i)=>{const x=w-(now-p.time)/span*w,y=mid-p.value*h*.16;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.lineTo(w,mid);ctx.stroke()});requestAnimationFrame(drawTimeline)}
function showToast(message) { els.toast.textContent=message;els.toast.classList.add("show");clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>els.toast.classList.remove("show"),3800); }
init();
