(() => {
  const button=document.querySelector('#recordButton'), stage=document.querySelector('#stageLabel'), timer=document.querySelector('#timer');
  let socket,stream,context,source,worklet,processor,startedAt,timerId,active=false,floatQueue=[],queued=0,audioUrl=null,inputUrl=null,pcmChunks=[],sentSamples=0,playQueue=[],playing=false,dictationText='';

  button.addEventListener('click', async event => {
    event.stopImmediatePropagation();
    if(active) stopAndCommit(); else await startLive();
  }, true);

  async function startLive(){
    try{
      pcmChunks=[];sentSamples=0;playQueue=[];timeline.input=[];timeline.output=[];state.audioUrls.forEach(URL.revokeObjectURL);state.audioUrls=[];document.querySelector('#translationFeed').innerHTML='<div class="empty">Chưa có segment</div>';document.querySelector('#textTTFT').textContent='TTFT —';document.querySelector('#audioTTFA').textContent='TTFA —'; updateMonitor(null,'Đang thu PCM16 · 16 kHz');
      if(!navigator.mediaDevices?.getUserMedia)throw new Error('Microphone cần HTTPS hoặc localhost');
      stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      socket=new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/api/live`);
      socket.binaryType='arraybuffer'; socket.onmessage=handleEvent;
      socket.onerror=()=>fail('WebSocket live bị lỗi. Kiểm tra reverse proxy có hỗ trợ Upgrade.');
      socket.onclose=event=>{ if(active){fail(`Live session đóng bất ngờ (${event.code}).`); cleanup(false);} };
      await opened(socket,5000);
      socket.send(JSON.stringify({type:'session.configure',source:document.querySelector('#sourceLanguage').value,target:document.querySelector('#targetLanguage').value,speed:Number(document.querySelector('#speed').value)}));

      const AudioContextClass=window.AudioContext||window.webkitAudioContext;
      if(!AudioContextClass)throw new Error('Trình duyệt không hỗ trợ Web Audio API');
      context=new AudioContextClass({latencyHint:'interactive'}); await context.resume();
      source=context.createMediaStreamSource(stream);
      if(context.audioWorklet&&window.AudioWorkletNode){
        try{
          await context.audioWorklet.addModule('/static/pcm-worklet.js');
          worklet=new AudioWorkletNode(context,'pcm-collector');worklet.port.onmessage=event=>enqueue(event.data);source.connect(worklet);worklet.connect(context.destination);
        }catch(error){setupProcessorFallback(error);}
      }else setupProcessorFallback();
      active=true; startedAt=Date.now(); timerId=setInterval(updateTimer,200);
      button.classList.add('recording'); button.querySelector('b').textContent='Dừng stream';
      stage.textContent='Đang chờ giọng nói · OmniVAD';
    }catch(error){ fail(`Không bắt đầu được live mode: ${error.message}`); cleanup(); }
  }

  function setupProcessorFallback(reason){
    processor=context.createScriptProcessor(4096,1,1);processor.onaudioprocess=event=>enqueue(event.inputBuffer.getChannelData(0).slice());source.connect(processor);processor.connect(context.destination);
    updateMonitor(null,`PCM16 · 16 kHz · compatibility capture${reason?' (AudioWorklet unavailable)':''}`);
  }

  function enqueue(samples){
    if(!active||socket?.readyState!==WebSocket.OPEN)return;
    floatQueue.push(samples); queued+=samples.length;
    if(queued<4096)return;
    const joined=new Float32Array(queued); let offset=0;
    for(const part of floatQueue){joined.set(part,offset);offset+=part.length;}
    floatQueue=[];queued=0; const pcm=downsample(joined,context.sampleRate,16000);let energy=0;for(const v of pcm)energy+=(v/32768)**2;recordTimeline('input',Math.sqrt(energy/Math.max(1,pcm.length))*5);pcmChunks.push(pcm.slice());sentSamples+=pcm.length;socket.send(pcm.buffer); updateMonitor(null,`PCM16 · 16 kHz · ${(sentSamples/16000).toFixed(1)}s đã gửi`);
  }
  function stopAndCommit(){ flush(); if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify({type:'input.commit'})); stage.textContent='Đang hoàn tất bản dịch';stopCapture(); }
  function flush(){if(!queued)return;const joined=new Float32Array(queued);let o=0;for(const p of floatQueue){joined.set(p,o);o+=p.length;}floatQueue=[];queued=0;const pcm=downsample(joined,context.sampleRate,16000);pcmChunks.push(pcm.slice());sentSamples+=pcm.length;if(socket?.readyState===WebSocket.OPEN)socket.send(pcm.buffer);}

  function handleEvent(message){
    const event=JSON.parse(message.data);
    if(event.type==='session.ready')stage.textContent=`Live ready · ${event.vad}`;
    if(event.type==='audio.received'){stage.textContent=event.speech?'Đang nghe · speech detected':`Đang chờ giọng nói · ${event.seconds.toFixed(1)}s`;document.querySelector('#inputMeta').textContent=`PCM16 · ${event.seconds.toFixed(1)}s · RMS ${event.rms} · peak ${event.peak}`;}
    if(event.type==='vad.speech_start'){dictationText='';stage.textContent=event.fallback?'Đang nghe · energy fallback':'Đang nghe · speech detected';renderDictation('','',false,'Listening');}
    if(event.type==='vad.speech_end')stage.textContent='Hết lượt nói · đang dịch · stream vẫn mở';
    if(event.type==='asr.dictation.delta'){if(!(event.text.length<dictationText.length&&dictationText.startsWith(event.text))){dictationText=event.text;renderDictation(event.text,event.stable_text,false,'Dictating');}stage.textContent='ASR dictation · nhận từng token';}
    if(event.type==='translation.update'){const sourceText=event.final?event.transcript:event.stable_transcript;const translatedText=event.final?event.translation:event.stable_translation;if(event.final){dictationText=event.transcript;renderDictation(event.transcript,event.stable_transcript,true);}if(translatedText)setText('#translation',translatedText);if(event.final||(sourceText&&translatedText))updateFeed(sourceText,translatedText,event.final);document.querySelector('#detectedLanguage').textContent=event.detected_language;document.querySelector('#latency').textContent=`${event.latency_ms} ms · ${event.final?'final':'dictation draft'}`;if(event.time_to_first_text_ms)document.querySelector('#textTTFT').textContent=`TTFT ${event.time_to_first_text_ms} ms`;stage.textContent=event.final?'Bản dịch đã commit · đang tạo giọng':(sourceText?'Bản dịch ổn định tạm thời':'Đang xác nhận hypothesis ASR');}
    if(event.type==='speech.started'){stage.textContent=`OmniVoice đang sinh segment ${event.index} · queue ${event.queue_wait_ms} ms`;document.querySelector('#audioTTFA').textContent=`TTFA đang chờ… · Q ${event.queue_wait_ms} ms`;}
    if(event.type==='speech.segment'){const bytes=Uint8Array.from(atob(event.audio),c=>c.charCodeAt(0));recordTimeline('output',.8);setTimeout(()=>recordTimeline('output',0),900);audioUrl=URL.createObjectURL(new Blob([bytes],{type:event.mime}));state.audioUrl=audioUrl;state.audioUrls.push(audioUrl);playQueue.push(audioUrl);document.querySelector('#outputPreview').src=audioUrl;document.querySelector('#playButton').disabled=false;document.querySelector('#audioTTFA').textContent=`TTFA ${event.time_to_first_audio_ms} ms`;stage.textContent=`Đang phát segment ${event.index} · TTS ${event.synthesis_ms} ms`;playNext();}
    if(event.type==='utterance.done')stage.textContent=`Đã cắt lượt ${event.index} · tiếp tục nghe`;
    if(event.type==='utterance.discarded'){stage.textContent=`Bỏ qua non-speech ${event.voiced_ms} ms · tiếp tục nghe`;renderDictation('','',false,'Discarded');}
    if(event.type==='speech.done'){stage.textContent=`Đã tạo ${event.segments} audio segment`;}
    if(event.type==='session.done'){stage.textContent=`Đã dừng stream · ${event.utterances||0} lượt nói`;cleanup();}
    if(event.type==='error'){fail(event.message);cleanup();}
  }

  function playNext(){
    if(playing||!playQueue.length)return;
    const url=playQueue.shift(),audio=new Audio(url);playing=true;
    audio.onended=()=>{playing=false;playNext();};
    audio.onerror=()=>{playing=false;stage.textContent='Audio segment lỗi decode · dùng nút Nghe để thử lại';};
    audio.play().catch(error=>{
      playing=false;playQueue.unshift(url);
      stage.textContent='Audio đã sẵn sàng · bấm Nghe bản dịch';
      const toast=document.querySelector('#toast');toast.textContent=`Trình duyệt chặn tự phát audio (${error.name}). Bấm Nghe bản dịch.`;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),5000);
    });
  }


  function buildPCMPreview(){if(!sentSamples)return;const pcm=new Int16Array(sentSamples);let offset=0;for(const part of pcmChunks){pcm.set(part,offset);offset+=part.length;}const blob=wavBlob(pcm,16000);updateMonitor(blob,`PCM16 · 16 kHz · ${(sentSamples/16000).toFixed(1)}s · ${(blob.size/1024).toFixed(0)} KB`);}
  function wavBlob(pcm,rate){const buffer=new ArrayBuffer(44+pcm.byteLength),view=new DataView(buffer);const text=(o,v)=>[...v].forEach((c,i)=>view.setUint8(o+i,c.charCodeAt(0)));text(0,'RIFF');view.setUint32(4,36+pcm.byteLength,true);text(8,'WAVE');text(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,rate,true);view.setUint32(28,rate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);text(36,'data');view.setUint32(40,pcm.byteLength,true);new Int16Array(buffer,44).set(pcm);return new Blob([buffer],{type:'audio/wav'});}
  function updateMonitor(blob,meta){const monitor=document.querySelector('#inputMonitor'),preview=document.querySelector('#inputPreview'),download=document.querySelector('#downloadInput');document.querySelector('#inputMeta').textContent=meta;monitor.classList.toggle('has-audio',!!blob||sentSamples>0);if(!blob)return;if(inputUrl)URL.revokeObjectURL(inputUrl);inputUrl=URL.createObjectURL(blob);preview.src=inputUrl;download.href=inputUrl;download.hidden=false;}

  function downsample(input,from,to){if(from===to)return floatTo16(input);const ratio=from/to,length=Math.floor(input.length/ratio),out=new Int16Array(length);for(let i=0;i<length;i++){const a=Math.floor(i*ratio),b=Math.min(input.length,Math.floor((i+1)*ratio));let sum=0;for(let j=a;j<b;j++)sum+=input[j];const v=Math.max(-1,Math.min(1,sum/Math.max(1,b-a)));out[i]=v<0?v*32768:v*32767;}return out;}
  function floatTo16(input){const out=new Int16Array(input.length);input.forEach((v,i)=>{v=Math.max(-1,Math.min(1,v));out[i]=v<0?v*32768:v*32767});return out;}
  function opened(ws,ms){return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('WebSocket timeout')),ms);ws.addEventListener('open',()=>{clearTimeout(timeout);resolve();},{once:true});ws.addEventListener('error',()=>{clearTimeout(timeout);reject(new Error('WebSocket connection failed'));},{once:true});});}
  function setText(selector,text){const el=document.querySelector(selector);el.textContent=text;el.classList.remove('empty');}
  function renderDictation(raw='',stable='',final=false,label){const el=document.querySelector('#transcript'),stateTag=document.querySelector('#dictationState');el.innerHTML='';el.classList.remove('empty');if(final){const locked=document.createElement('span');locked.className='dictation-final';locked.textContent=raw||'—';el.appendChild(locked);stateTag.textContent='Final';return;}const confirmed=raw.startsWith(stable)?stable:'';const hypothesis=raw.startsWith(stable)?raw.slice(stable.length):raw;if(confirmed){const span=document.createElement('span');span.className='dictation-stable';span.textContent=confirmed;el.appendChild(span);}if(hypothesis){const span=document.createElement('span');span.className='dictation-hypothesis';span.textContent=hypothesis;el.appendChild(span);}if(!raw){const waiting=document.createElement('span');waiting.className='dictation-hypothesis';waiting.textContent='Đang nghe…';el.appendChild(waiting);}const caret=document.createElement('i');caret.className='dictation-caret';caret.setAttribute('aria-hidden','true');el.appendChild(caret);stateTag.textContent=label||'Draft';}
  function updateTimer(){const s=Math.floor((Date.now()-startedAt)/1000);timer.textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;}
  function stopCapture(){if(active){flush();buildPCMPreview();}active=false;clearInterval(timerId);worklet?.disconnect();worklet=null;if(processor){processor.onaudioprocess=null;processor.disconnect();processor=null;}source?.disconnect();source=null;stream?.getTracks().forEach(t=>t.stop());stream=null;context?.close();context=null;button.classList.remove('recording');button.querySelector('b').textContent='Bắt đầu stream';}
  function cleanup(close=true){stopCapture();if(close&&socket?.readyState===WebSocket.OPEN)socket.close(1000);}
  function fail(message){const toast=document.querySelector('#toast');toast.textContent=message;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),5000);stage.textContent='Live bị lỗi · xem thông báo';}
  document.querySelector('#audioFile').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;if(inputUrl)URL.revokeObjectURL(inputUrl);inputUrl=URL.createObjectURL(file);document.querySelector('#inputPreview').src=inputUrl;document.querySelector('#inputMeta').textContent=`${file.type||'audio'} · ${(file.size/1024).toFixed(0)} KB`;document.querySelector('#inputMonitor').classList.add('has-audio');document.querySelector('#downloadInput').href=inputUrl;document.querySelector('#downloadInput').download=file.name;document.querySelector('#downloadInput').hidden=false;});
})();
