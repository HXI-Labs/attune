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
      --page: #faf9f6;
      --text: #1a1c1b;
      --muted: #6b706c;
      --line: #deded8;
      --accent: #176454;
      --recording: #a33b38;
      --focus: #2668c7;
      --laugh: #b96b16;
      --breath: #3973a8;
      --cough: #8a4e9c;
      --sigh: #367f70;
      --sob: #b4435c;
      --speech-style: #8a4d27;
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
      width: min(720px, calc(100% - 40px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }

    header {
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }

    .wordmark { margin: 0; font-size: 15px; font-weight: 700; letter-spacing: -0.01em; }

    .recorder {
      display: grid;
      justify-items: center;
      min-height: 300px;
      align-content: center;
      text-align: center;
    }

    h1 {
      margin: 0 0 8px;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: clamp(34px, 6vw, 48px);
      font-weight: 500;
      line-height: 1.08;
      letter-spacing: -0.035em;
    }

    .intro { margin: 0 0 26px; color: var(--muted); }
    button, summary, .file-link { -webkit-tap-highlight-color: transparent; }

    .record-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 148px;
      min-height: 48px;
      border: 0;
      border-radius: 9px;
      padding: 0 20px;
      background: var(--text);
      color: #fff;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      transition: background 120ms ease;
    }

    .record-button:hover:not(:disabled) { background: #333634; }
    .record-button:disabled { cursor: wait; opacity: 0.62; }
    .record-button.recording { background: var(--recording); }
    .status { min-height: 21px; margin: 13px 0 0; color: var(--muted); font-size: 12px; }
    .status.error { color: var(--recording); }

    .file-link {
      display: inline-block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      text-decoration: underline;
      text-underline-offset: 3px;
      cursor: pointer;
    }

    input[type="file"] {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }

    #results { display: none; }
    #results.visible { display: block; }
    .result-shell { border-top: 1px solid var(--line); padding: 38px 0 0; }

    .result-heading {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .transcript {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: clamp(29px, 5vw, 42px);
      font-weight: 500;
      line-height: 1.42;
      letter-spacing: -0.025em;
    }

    ruby.annotation {
      margin: 0 0.04em;
      color: var(--mark, var(--accent));
      text-decoration-line: underline;
      text-decoration-color: currentColor;
      text-decoration-thickness: 0.075em;
      text-underline-offset: 0.12em;
    }

    ruby.annotation rt {
      color: var(--mark, var(--accent));
      font: 700 10px/1.1 Inter, ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0.035em;
      text-transform: lowercase;
    }

    .event-token { font-size: 0.7em; letter-spacing: -0.01em; }
    .affect-summary { margin: 18px 0 0; color: var(--muted); font-size: 13px; }

    .details {
      margin-top: 24px;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }

    .details > summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 54px;
      color: var(--text);
      font-weight: 650;
      cursor: pointer;
      list-style: none;
    }

    .details > summary::-webkit-details-marker { display: none; }
    .details > summary::after {
      content: "+";
      color: var(--muted);
      font-size: 19px;
      font-weight: 400;
    }
    .details[open] > summary::after { content: "−"; }

    .detail-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 32px;
      padding: 4px 0 26px;
    }

    .detail-section + .detail-section { margin-top: 24px; }
    h2 { margin: 0 0 10px; font-size: 13px; }
    .evidence-list { margin: 0; padding-left: 18px; }
    .evidence-list li + li { margin-top: 6px; }
    .muted { color: var(--muted); font-size: 13px; }

    .affect-row {
      display: grid;
      grid-template-columns: 76px 1fr 38px;
      gap: 9px;
      align-items: center;
      margin: 8px 0;
      font-size: 12px;
    }

    progress { width: 100%; height: 5px; border: 0; background: #e3e2dc; }
    progress::-webkit-progress-bar { background: #e3e2dc; }
    progress::-webkit-progress-value { background: var(--accent); }
    progress::-moz-progress-bar { background: var(--accent); }
    .number { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
    dl {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 7px 13px;
      margin: 0;
      font-size: 13px;
    }
    dt { color: var(--muted); }
    dd { margin: 0; }
    audio { width: 100%; margin-top: 10px; }

    .raw-details { margin-top: 20px; }
    .raw-details summary { color: var(--muted); cursor: pointer; font-size: 12px; }

    pre {
      max-height: 380px;
      overflow: auto;
      margin: 10px 0 0;
      padding: 13px;
      border: 1px solid var(--line);
      background: #f2f1ed;
      font-size: 11px;
    }

    .interpretation-note { margin: 20px 0 0; color: var(--muted); font-size: 11px; }

    .record-button:focus-visible,
    .file-link:focus-within,
    summary:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--focus) 34%, transparent);
      outline-offset: 3px;
    }

    @media (max-width: 680px) {
      main { width: min(100% - 28px, 720px); padding-top: 20px; }
      .recorder { min-height: 270px; }
      .result-shell { padding-top: 30px; }
      .detail-grid { grid-template-columns: 1fr; gap: 24px; }
    }

    @media (prefers-reduced-motion: reduce) {
      .record-button { transition: none; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <p class="wordmark">Attune Cadence</p>
    </header>

    <section class="recorder" aria-labelledby="page-title">
      <h1 id="page-title">Record and transcribe.</h1>
      <p class="intro">Speak naturally for up to 30 seconds.</p>
      <button class="record-button" id="record-button" type="button">
        <span id="record-label">Record</span>
      </button>
      <p class="status" id="status" role="status" aria-live="polite">
        Audio is processed in memory and is not stored.
      </p>
      <label class="file-link" for="file-input">Upload audio</label>
      <input id="file-input" type="file" accept="audio/*,.wav">
    </section>

    <section id="results" aria-labelledby="transcript-heading">
      <div class="result-shell">
        <p class="result-heading" id="transcript-heading">Transcript</p>
        <p class="transcript" id="transcript"></p>
        <p class="affect-summary" id="affect-summary"></p>

        <details class="details">
          <summary>Details</summary>
          <div class="detail-grid">
            <div>
              <section class="detail-section">
                <h2>Evidence</h2>
                <ul class="evidence-list" id="events"></ul>
              </section>
              <section class="detail-section">
                <h2>Timing</h2>
                <dl id="timing"></dl>
                <audio id="audio-preview" controls hidden></audio>
              </section>
            </div>
            <div>
              <section class="detail-section">
                <h2>Affect</h2>
                <div id="affect"></div>
              </section>
              <section class="detail-section">
                <h2>Confidence</h2>
                <dl id="uncertainty"></dl>
              </section>
            </div>
          </div>
          <details class="raw-details">
            <summary>JSON</summary>
            <pre id="json"></pre>
          </details>
        </details>
        <p class="interpretation-note">Affect labels describe vocal delivery and may be wrong.</p>
      </div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("file-input");
    const recordButton = document.getElementById("record-button");
    const audioPreview = document.getElementById("audio-preview");

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

    function setRecordState(label, state = "") {
      element("record-label").textContent = label;
      recordButton.className = `record-button ${state}`;
    }

    function formatPercent(value) { return `${Math.round(value * 100)}%`; }
    function clearElement(target) { target.replaceChildren(); }

    function markColor(label) {
      if (label.includes("joy")) return "var(--laugh)";
      if (label.includes("distress")) return "var(--sob)";
      if (label.includes("anger")) return "var(--recording)";
      if (label.includes("fear")) return "var(--breath)";
      const colors = {
        laugh: "var(--laugh)",
        laughing_speech: "var(--laugh)",
        breath: "var(--breath)",
        cough: "var(--cough)",
        throat_clear: "var(--cough)",
        sigh: "var(--sigh)",
        sob: "var(--sob)",
        crying_speech: "var(--sob)",
      };
      return colors[label] || "var(--speech-style)";
    }

    function affectSpanLabel(span) {
      const strongest = Object.entries(span.categories).sort(
        (left, right) => right[1] - left[1],
      )[0];
      if (strongest?.[0] === "neutral") return null;
      if (!span.abstain && span.top_label) return `perceived ${span.top_label}`;
      if (strongest && strongest[1] >= 0.25) {
        return `possible ${strongest[0]} · uncertain`;
      }
      return "affect uncertain";
    }

    function downsample(samples, inputRate, outputRate = 16000) {
      if (inputRate === outputRate) return samples;
      const ratio = inputRate / outputRate;
      const output = new Float32Array(Math.round(samples.length / ratio));
      for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
        const start = Math.round(outputIndex * ratio);
        const end = Math.min(samples.length, Math.round((outputIndex + 1) * ratio));
        let sum = 0;
        for (let inputIndex = start; inputIndex < end; inputIndex += 1) sum += samples[inputIndex];
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

    function appendDefinition(list, term, description) {
      const termNode = document.createElement("dt");
      const descriptionNode = document.createElement("dd");
      termNode.textContent = term;
      descriptionNode.textContent = description;
      list.append(termNode, descriptionNode);
    }

    function appendListEntry(list, text) {
      const entry = document.createElement("li");
      entry.textContent = text;
      list.append(entry);
    }

    function annotation(baseText, labels, className = "") {
      const ruby = document.createElement("ruby");
      const base = document.createElement("span");
      const caption = document.createElement("rt");
      ruby.className = `annotation ${className}`;
      ruby.style.setProperty("--mark", markColor(labels[0]));
      base.textContent = baseText;
      caption.textContent = labels.map((label) => label.replaceAll("_", " ")).join(" · ");
      ruby.append(base, caption);
      return ruby;
    }

    function renderTranscript(attune) {
      const transcript = element("transcript");
      clearElement(transcript);
      const words = attune.transcript.words || [];
      if (words.length === 0) {
        transcript.textContent = attune.transcript.text || "No speech was transcribed.";
        return;
      }

      const eventsAfterWord = new Map();
      for (const event of attune.events) {
        let wordId = event.after_word_id;
        if (!wordId && event.start_ms !== null) {
          const previous = words.filter((word) => word.end_ms <= event.start_ms).at(-1);
          wordId = previous ? previous.id : words[0].id;
        }
        const existing = eventsAfterWord.get(wordId) || [];
        existing.push(event);
        eventsAfterWord.set(wordId, existing);
      }

      let activeLabels = [];
      let activeWords = [];
      const appendSeparated = (node) => {
        if (transcript.childNodes.length) transcript.append(" ");
        transcript.append(node);
      };
      const flushWords = () => {
        if (!activeWords.length) return;
        const phrase = activeWords.join(" ");
        appendSeparated(
          activeLabels.length
            ? annotation(phrase, activeLabels)
            : document.createTextNode(phrase),
        );
        activeWords = [];
      };

      words.forEach((word) => {
        const styleLabels = attune.styles
          .filter((style) => style.temporal_scope === "utterance"
            || (style.start_ms < word.end_ms && style.end_ms > word.start_ms))
          .map((style) => style.label);
        const affectSpans = attune.affect_spans?.length
          ? attune.affect_spans
          : [attune.affect];
        const affectLabels = affectSpans
          .filter((span) => span.start_ms < word.end_ms && span.end_ms > word.start_ms)
          .map(affectSpanLabel)
          .filter(Boolean);
        const labels = [...styleLabels, ...affectLabels];
        if (JSON.stringify(labels) !== JSON.stringify(activeLabels)) {
          flushWords();
          activeLabels = labels;
        }
        activeWords.push(word.text);
        for (const event of eventsAfterWord.get(word.id) || []) {
          flushWords();
          const eventText = `[${event.label.replaceAll("_", " ")}]`;
          appendSeparated(annotation(eventText, [event.label], "event-token"));
        }
      });
      flushWords();
    }

    function renderAnalysis(analysis) {
      const attune = analysis.result;
      const affect = attune.affect;
      renderTranscript(attune);
      element("affect-summary").textContent = !affect.abstain && affect.top_label
        ? `${affect.top_label} · ${formatPercent(affect.top_label_confidence)}`
        : "Affect uncertain";

      const eventList = element("events");
      clearElement(eventList);
      for (const style of attune.styles) {
        const label = style.label.replaceAll("_", " ");
        appendListEntry(
          eventList,
          `${label} · ${formatPercent(style.confidence)} · ${style.start_ms}–${style.end_ms} ms`,
        );
      }
      for (const event of attune.events) {
        const timing = event.start_ms === null ? "" : ` · ${event.start_ms}–${event.end_ms} ms`;
        appendListEntry(
          eventList,
          `${event.label.replaceAll("_", " ")} · ${formatPercent(event.confidence)}${timing}`,
        );
      }
      if (eventList.childElementCount === 0) {
        appendListEntry(eventList, "No supported event or style was detected.");
      }

      const timing = element("timing");
      clearElement(timing);
      appendDefinition(
        timing,
        "Audio",
        `${(analysis.timing.audio_duration_ms / 1000).toFixed(1)} s`,
      );
      appendDefinition(timing, "Processing", `${analysis.timing.processing_ms} ms`);
      appendDefinition(timing, "Real-time factor", analysis.timing.real_time_factor.toFixed(3));

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
      appendDefinition(uncertainty, "Decision", affect.abstain ? "Abstained" : "Returned");
      appendDefinition(
        uncertainty,
        "Reason",
        affect.abstention_reason || "The top category passed the calibrated threshold.",
      );
      appendDefinition(
        uncertainty,
        "Out of distribution",
        attune.uncertainty.out_of_distribution_available === false
          ? "Unavailable"
          : formatPercent(attune.uncertainty.out_of_distribution_probability),
      );

      element("json").textContent = JSON.stringify(analysis, null, 2);
      element("results").classList.add("visible");
      element("results").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    async function analyseAudio(wav) {
      setPreview(wav);
      recordButton.disabled = true;
      setRecordState("Working…");
      setStatus("Analysing audio.", "busy");
      try {
        const form = new FormData();
        form.append("audio", wav, "attune-input.wav");
        const response = await fetch("/v1/analyse", { method: "POST", body: form });
        const analysis = await response.json();
        if (!response.ok) throw new Error(analysis.detail || "Analysis failed.");
        renderAnalysis(analysis);
        setStatus("Done.");
        setRecordState("Record again");
      } catch (error) {
        setStatus(error.message, "error");
        setRecordState("Try again");
      } finally {
        recordButton.disabled = false;
      }
    }

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
      const durationSeconds = (Date.now() - recordingStartedAt) / 1000;
      if (durationSeconds < 0.5) throw new Error("Record at least half a second of audio.");
      await analyseAudio(encodePcm16Wav(downsample(joined, sampleRate)));
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
      setRecordState("Stop", "recording");
      setStatus("Recording…", "busy");
    }

    recordButton.addEventListener("click", async () => {
      try {
        if (recorderNode !== null) await stopRecording();
        else await startRecording();
      } catch (error) {
        if (mediaStream !== null) mediaStream.getTracks().forEach((track) => track.stop());
        recorderNode = null;
        microphoneSource = null;
        mediaStream = null;
        setRecordState("Try again");
        recordButton.disabled = false;
        setStatus(`Microphone unavailable: ${error.message}`, "error");
      }
    });

    fileInput.addEventListener("change", async () => {
      if (!fileInput.files[0]) return;
      recordButton.disabled = true;
      setStatus("Preparing audio.", "busy");
      try {
        await analyseAudio(await normalizeAudio(fileInput.files[0]));
      } catch (error) {
        setStatus(error.message, "error");
        setRecordState("Try again");
        recordButton.disabled = false;
      } finally {
        fileInput.value = "";
      }
    });
  </script>
</body>
</html>
"""
