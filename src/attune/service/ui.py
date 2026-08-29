"""Browser interface for local Attune evaluation."""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attune Cadence</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f2f0e9;
      --surface: #ffffff;
      --surface-muted: #e9e6dd;
      --border: #d3d0c7;
      --text: #202220;
      --muted: #646862;
      --accent: #28675b;
      --accent-hover: #1f554c;
      --danger: #a53b3b;
      --focus: #1f6feb;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--page);
      color: var(--text);
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }

    main {
      width: min(960px, calc(100% - 32px));
      margin: 0 auto;
      padding: 44px 0 64px;
    }

    header { max-width: 680px; margin-bottom: 28px; }

    .project-label {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: clamp(34px, 6vw, 48px);
      line-height: 1.08;
      letter-spacing: -0.035em;
    }

    header p { margin: 12px 0 0; color: var(--muted); font-size: 17px; }

    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
    }

    .input-panel { padding: 22px; }

    .drop-area {
      min-height: 132px;
      display: grid;
      place-items: center;
      border: 1px dashed #999d96;
      border-radius: 8px;
      cursor: pointer;
      text-align: center;
    }

    .drop-area:hover,
    .drop-area.dragging { border-color: var(--accent); background: #f4f8f6; }
    .drop-area strong { display: block; margin-bottom: 4px; font-size: 16px; }
    .muted, .drop-area span { color: var(--muted); font-size: 13px; }

    input[type="file"] {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }

    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }

    button {
      min-height: 42px;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 9px 16px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:hover:not(:disabled) { background: var(--surface-muted); }
    button.primary { border-color: var(--accent); background: var(--accent); color: #ffffff; }
    button.primary:hover:not(:disabled) { background: var(--accent-hover); }
    button.recording { border-color: var(--danger); background: var(--danger); color: #ffffff; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }

    button:focus-visible,
    .drop-area:focus-within,
    summary:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--focus) 35%, transparent);
      outline-offset: 2px;
    }

    .status { min-height: 24px; margin: 12px 0 0; color: var(--muted); }
    .status.error { color: var(--danger); }

    .status.busy::after {
      content: "";
      display: inline-block;
      width: 6px;
      height: 6px;
      margin-left: 8px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 0.8s infinite alternate;
    }

    @keyframes pulse { to { opacity: 0.2; } }
    audio { width: 100%; margin-top: 14px; }
    #results { display: none; margin-top: 20px; }
    #results.visible { display: block; }
    .section { padding: 22px; border-bottom: 1px solid var(--border); }
    .section:last-child { border-bottom: 0; }
    h2, h3 { margin: 0 0 12px; line-height: 1.25; }
    h2 { font-size: 15px; letter-spacing: 0.04em; text-transform: uppercase; }
    h3 { font-size: 17px; }

    .transcript {
      margin: 0 0 14px;
      font-size: clamp(24px, 4vw, 34px);
      line-height: 1.25;
      letter-spacing: -0.02em;
    }

    .summary { margin: 0 0 12px; color: var(--accent); font-weight: 700; }
    .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }

    .affect-row {
      display: grid;
      grid-template-columns: 82px 1fr 42px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }

    progress {
      width: 100%;
      height: 8px;
      border: 0;
      border-radius: 0;
      background: var(--surface-muted);
    }

    progress::-webkit-progress-bar { background: var(--surface-muted); }
    progress::-webkit-progress-value { background: var(--accent); }
    progress::-moz-progress-bar { background: var(--accent); }

    .number {
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .evidence-list { margin: 0; padding-left: 20px; }
    .evidence-list li + li { margin-top: 7px; }
    dl { display: grid; grid-template-columns: max-content 1fr; gap: 7px 14px; margin: 0; }
    dt { color: var(--muted); }
    dd { margin: 0; }
    details { padding-top: 2px; }
    summary { cursor: pointer; font-weight: 700; }

    pre {
      max-height: 460px;
      overflow: auto;
      margin: 14px 0 0;
      padding: 14px;
      border: 1px solid var(--border);
      background: #f6f5f1;
      color: #30342f;
      font-size: 12px;
    }

    footer {
      max-width: 720px;
      margin: 20px 0 0;
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 700px) {
      main { padding-top: 28px; }
      .columns { grid-template-columns: 1fr; }
      .input-panel, .section { padding: 17px; }
    }

    @media (prefers-reduced-motion: reduce) {
      .status.busy::after { animation: none; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <p class="project-label">Project Attune</p>
      <h1>Attune Cadence</h1>
      <p>Test transcription, localized vocal events, and perceived affect from one voice clip.</p>
    </header>

    <section class="panel input-panel" aria-labelledby="audio-heading">
      <h2 id="audio-heading">Audio input</h2>
      <label class="drop-area" id="drop-area" for="file-input">
        <span>
          <strong id="file-label">Drop an audio file here</strong>
          Choose a file instead · 0.5–30 seconds
        </span>
      </label>
      <input id="file-input" type="file" accept="audio/*,.wav">
      <div class="actions">
        <button id="record-button" type="button">Record</button>
        <button class="primary" id="analyse-button" type="button" disabled>Analyse audio</button>
      </div>
      <audio id="audio-preview" controls hidden></audio>
      <p class="status" id="status" role="status" aria-live="polite">
        Choose a file or record from the microphone.
      </p>
    </section>

    <section class="panel" id="results" aria-labelledby="transcript-heading">
      <div class="section">
        <h2 id="transcript-heading">Transcript</h2>
        <p class="transcript" id="transcript"></p>
        <p class="summary" id="summary"></p>
        <ul class="evidence-list" id="events"></ul>
        <p class="muted" id="timing"></p>
      </div>

      <div class="section columns">
        <div><h3>Perceived affect</h3><div id="affect"></div></div>
        <div><h3>Uncertainty</h3><dl id="uncertainty"></dl></div>
      </div>

      <div class="section">
        <details><summary>Structured JSON</summary><pre id="json"></pre></details>
      </div>
    </section>

    <footer>
      Perceived vocal expression is uncertain evidence, not a verified internal state,
      diagnosis, or safety decision. Audio is processed in memory and is not stored.
    </footer>
  </main>

  <script>
    const fileInput = document.getElementById("file-input");
    const dropArea = document.getElementById("drop-area");
    const analyseButton = document.getElementById("analyse-button");
    const recordButton = document.getElementById("record-button");
    const audioPreview = document.getElementById("audio-preview");

    let selectedWav = null;
    let previewUrl = null;
    let recorderNode = null;
    let microphoneSource = null;
    let mediaStream = null;
    let audioChunks = [];
    let audioContext = null;
    let recordingStartedAt = 0;

    function element(id) { return document.getElementById(id); }

    function setStatus(message, kind = "") {
      const status = element("status");
      status.textContent = message;
      status.className = `status ${kind}`;
    }

    function formatPercent(value) { return `${Math.round(value * 100)}%`; }
    function clearElement(target) { target.replaceChildren(); }

    function downsample(samples, inputRate, outputRate = 16000) {
      if (inputRate === outputRate) return samples;
      const ratio = inputRate / outputRate;
      const output = new Float32Array(Math.round(samples.length / ratio));
      for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
        const start = Math.round(outputIndex * ratio);
        const end = Math.min(samples.length, Math.round((outputIndex + 1) * ratio));
        let sum = 0;
        for (let inputIndex = start; inputIndex < end; inputIndex += 1) {
          sum += samples[inputIndex];
        }
        output[outputIndex] = sum / Math.max(1, end - start);
      }
      return output;
    }

    function encodePcm16Wav(samples, sampleRate = 16000) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      const writeText = (offset, text) => {
        [...text].forEach((character, index) => {
          view.setUint8(offset + index, character.charCodeAt(0));
        });
      };
      writeText(0, "RIFF");
      view.setUint32(4, 36 + samples.length * 2, true);
      writeText(8, "WAVE");
      writeText(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeText(36, "data");
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (const sample of samples) {
        const bounded = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, bounded < 0 ? bounded * 32768 : bounded * 32767, true);
        offset += 2;
      }
      return new Blob([buffer], { type: "audio/wav" });
    }

    async function normalizeAudio(source) {
      setStatus("Preparing 16 kHz mono audio.", "busy");
      const decodingContext = new AudioContext();
      try {
        const bytes = (await source.arrayBuffer()).slice(0);
        const decoded = await decodingContext.decodeAudioData(bytes);
        const mono = new Float32Array(decoded.length);
        for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
          const channelSamples = decoded.getChannelData(channel);
          for (let index = 0; index < channelSamples.length; index += 1) {
            mono[index] += channelSamples[index] / decoded.numberOfChannels;
          }
        }
        const converted = downsample(mono, decoded.sampleRate);
        if (converted.length < 8000 || converted.length > 480000) {
          throw new Error("Audio must be between 0.5 and 30 seconds.");
        }
        return encodePcm16Wav(converted);
      } finally {
        await decodingContext.close();
      }
    }

    function setPreview(wav) {
      if (previewUrl !== null) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(wav);
      audioPreview.src = previewUrl;
      audioPreview.hidden = false;
    }

    async function selectAudio(source, label = source.name) {
      try {
        selectedWav = await normalizeAudio(source);
        element("file-label").textContent = label;
        analyseButton.disabled = false;
        setPreview(selectedWav);
        setStatus("Audio is ready to analyse.");
      } catch (error) {
        selectedWav = null;
        analyseButton.disabled = true;
        setStatus(error.message, "error");
      }
    }

    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) selectAudio(fileInput.files[0]);
    });

    for (const eventName of ["dragenter", "dragover"]) {
      dropArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropArea.classList.add("dragging");
      });
    }

    for (const eventName of ["dragleave", "drop"]) {
      dropArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropArea.classList.remove("dragging");
      });
    }

    dropArea.addEventListener("drop", (event) => {
      if (event.dataTransfer.files[0]) selectAudio(event.dataTransfer.files[0]);
    });

    async function stopRecording() {
      recorderNode.disconnect();
      microphoneSource.disconnect();
      mediaStream.getTracks().forEach((track) => track.stop());
      const sampleCount = audioChunks.reduce((total, chunk) => total + chunk.length, 0);
      const joined = new Float32Array(sampleCount);
      let offset = 0;
      for (const chunk of audioChunks) {
        joined.set(chunk, offset);
        offset += chunk.length;
      }
      const sampleRate = audioContext.sampleRate;
      await audioContext.close();
      recorderNode = null;
      microphoneSource = null;
      mediaStream = null;
      audioContext = null;
      recordButton.textContent = "Record";
      recordButton.className = "";
      const durationSeconds = (Date.now() - recordingStartedAt) / 1000;
      const label = `Microphone recording · ${durationSeconds.toFixed(1)} seconds`;
      await selectAudio(encodePcm16Wav(downsample(joined, sampleRate)), label);
    }

    async function startRecording() {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext();
      microphoneSource = audioContext.createMediaStreamSource(mediaStream);
      recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
      audioChunks = [];
      recorderNode.onaudioprocess = (event) => {
        audioChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        if (Date.now() - recordingStartedAt > 30000) recordButton.click();
      };
      microphoneSource.connect(recorderNode);
      recorderNode.connect(audioContext.destination);
      recordingStartedAt = Date.now();
      recordButton.textContent = "Stop recording";
      recordButton.className = "recording";
      analyseButton.disabled = true;
      setStatus("Recording. Speak naturally.", "busy");
    }

    recordButton.addEventListener("click", async () => {
      try {
        if (recorderNode !== null) await stopRecording();
        else await startRecording();
      } catch (error) {
        setStatus(`Microphone unavailable: ${error.message}`, "error");
      }
    });

    function appendListEntry(list, text) {
      const entry = document.createElement("li");
      entry.textContent = text;
      list.append(entry);
    }

    function appendDefinition(list, term, description) {
      const termNode = document.createElement("dt");
      const descriptionNode = document.createElement("dd");
      termNode.textContent = term;
      descriptionNode.textContent = description;
      list.append(termNode, descriptionNode);
    }

    function renderAnalysis(analysis) {
      const attune = analysis.result;
      const affect = attune.affect;
      const styleSummary = attune.styles.map((style) => style.label.replaceAll("_", " "));
      const affectSummary = !affect.abstain && affect.top_label
        ? `perceived ${affect.top_label} ${formatPercent(affect.top_label_confidence)}`
        : "affect uncertain";
      styleSummary.push(affectSummary);

      element("transcript").textContent = attune.transcript.text || "No speech was transcribed.";
      element("summary").textContent = `[${styleSummary.join("; ")}] ${attune.transcript.text}`;

      const eventList = element("events");
      clearElement(eventList);
      for (const style of attune.styles) {
        const label = style.label.replaceAll("_", " ");
        appendListEntry(eventList, `${label} · ${formatPercent(style.confidence)}`);
      }
      for (const event of attune.events) {
        const timing = event.start_ms === null ? "" : ` · ${event.start_ms}–${event.end_ms} ms`;
        const label = event.label.replaceAll("_", " ");
        appendListEntry(eventList, `${label} · ${formatPercent(event.confidence)}${timing}`);
      }
      if (eventList.childElementCount === 0) {
        appendListEntry(eventList, "No supported event or style was detected.");
      }

      element("timing").textContent = [
        `${analysis.timing.audio_duration_ms} ms audio`,
        `${analysis.timing.processing_ms} ms processing`,
        `real-time factor ${analysis.timing.real_time_factor.toFixed(3)}`,
      ].join(" · ");

      const affectPanel = element("affect");
      clearElement(affectPanel);
      const categories = Object.entries(affect.categories).sort(
        (left, right) => right[1] - left[1],
      );
      for (const [label, probability] of categories) {
        const row = document.createElement("div");
        const name = document.createElement("span");
        const bar = document.createElement("progress");
        const number = document.createElement("span");
        row.className = "affect-row";
        name.textContent = label;
        bar.max = 1;
        bar.value = probability;
        bar.setAttribute("aria-label", `${label} ${formatPercent(probability)}`);
        number.className = "number";
        number.textContent = formatPercent(probability);
        row.append(name, bar, number);
        affectPanel.append(row);
      }

      const uncertainty = element("uncertainty");
      clearElement(uncertainty);
      const decision = affect.abstain ? "Abstained" : "Interpretation returned";
      const reason = affect.abstention_reason
        || "The top category passed the calibrated threshold.";
      appendDefinition(uncertainty, "Decision", decision);
      appendDefinition(uncertainty, "Reason", reason);
      appendDefinition(
        uncertainty,
        "OOD probability",
        formatPercent(attune.uncertainty.out_of_distribution_probability),
      );
      appendDefinition(uncertainty, "Warning", attune.uncertainty.interpretation_warning);

      element("json").textContent = JSON.stringify(analysis, null, 2);
      element("results").classList.add("visible");
    }

    analyseButton.addEventListener("click", async () => {
      if (selectedWav === null) return;
      analyseButton.disabled = true;
      setStatus("Analysing audio.", "busy");
      try {
        const form = new FormData();
        form.append("audio", selectedWav, "attune-input.wav");
        const response = await fetch("/v1/analyse", { method: "POST", body: form });
        const analysis = await response.json();
        if (!response.ok) throw new Error(analysis.detail || "Analysis failed.");
        renderAnalysis(analysis);
        setStatus("Analysis complete.");
      } catch (error) {
        setStatus(error.message, "error");
      } finally {
        analyseButton.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
