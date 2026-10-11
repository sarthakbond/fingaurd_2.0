// FinGuard 2.0 Frontend Controller
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSystemHealth();
  initMediaScan();
  initMessageScan();
  initAppScan();
  initAlertsFeed();
  initDossierGenerator();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.dataset.target;
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// System Health & Engine Monitor
async function initSystemHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const statusDot = document.getElementById("header-status-dot");
    const statusText = document.getElementById("header-status-text");
    if (data.status === "HEALTHY") {
      if (statusText) statusText.textContent = "Phase 1: Foundation Active (Engines Ready)";
      if (statusDot) statusDot.style.background = "var(--accent-emerald)";
    }
  } catch (err) {
    console.warn("Health check error:", err);
    const statusText = document.getElementById("header-status-text");
    if (statusText) statusText.textContent = "Connecting to Backend...";
  }
}

// Requirement 1: Deepfake Media Forensic Engine
function initMediaScan() {
  const fileInput = document.getElementById("media-file-input");
  const dropZone = document.getElementById("media-drop-zone");
  const scanBtn = document.getElementById("media-scan-btn");
  const urlInput = document.getElementById("media-url-input");
  const resultCard = document.getElementById("media-result-card");

  // Presets
  document.querySelectorAll(".media-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.preset;
      if (type === "deepfake_audio") {
        urlInput.value = "https://samples.finguard.internal/cloned_deepfake_advisory_audio.wav";
      } else if (type === "authentic_audio") {
        urlInput.value = "https://samples.finguard.internal/authentic_market_update.mp3";
      } else if (type === "deepfake_video") {
        urlInput.value = "https://samples.finguard.internal/manipulated_finfluencer_video.mp4";
      } else {
        urlInput.value = "https://samples.finguard.internal/verified_executive_broadcast.mp4";
      }
    });
  });

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleMediaFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        handleMediaFileUpload(fileInput.files[0]);
      }
    });
  }

  if (scanBtn) {
    scanBtn.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (!url) return alert("Please provide a media URL or drop a media file.");

      setLoading(scanBtn, true);
      try {
        const res = await fetch("/api/scan/media", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ media_url: url })
        });
        const data = await res.json();
        renderMediaResult(data);
      } catch (err) {
        alert("Scan failed: " + err.message);
      } finally {
        setLoading(scanBtn, false);
      }
    });
  }

  async function handleMediaFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sampling_fps", "2.0");
    formData.append("extract_audio", "true");

    const btn = document.getElementById("media-upload-btn");
    setLoading(btn, true);

    try {
      const res = await fetch("/api/scan/media-file", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      renderMediaResult(data);
    } catch (err) {
      alert("File upload analysis failed: " + err.message);
    } finally {
      setLoading(btn, false);
    }
  }

  function renderMediaResult(data) {
    resultCard.style.display = "block";
    const badge = document.getElementById("media-verdict-badge");
    const vScoreEl = document.getElementById("media-visual-score");
    const vBarEl = document.getElementById("media-visual-bar");
    const aScoreEl = document.getElementById("media-acoustic-score");
    const aBarEl = document.getElementById("media-acoustic-bar");
    const timelineEl = document.getElementById("media-timeline-scrubber");
    const indicatorsEl = document.getElementById("media-indicators-list");
    const benchmarksEl = document.getElementById("media-benchmarks-list");

    // Verdict Badge
    if (data.status && data.status.startsWith("INFERENCE_")) {
      badge.className = "verdict-badge verdict-warning";
      badge.textContent = `UNVERIFIED: ${data.status.replace('_', ' ')}`;
    } else if (data.is_deepfake || data.spoken_advisory_verdict === "CRITICAL_FRAUD") {
      badge.className = "verdict-badge verdict-critical";
      badge.textContent = "CRITICAL: MANIPULATED DEEPFAKE / FRAUD";
    } else if (data.spoken_advisory_verdict === "SUSPICIOUS") {
      badge.className = "verdict-badge verdict-warning";
      badge.textContent = "SUSPICIOUS: ADVISORY FRAUD DETECTED";
    } else {
      badge.className = "verdict-badge verdict-safe";
      badge.textContent = "SAFE: AUTHENTIC MEDIA";
    }

    // Visual Meter
    const vScore = Math.round((data.visual_manipulation_score || 0) * 100);
    vScoreEl.textContent = `${vScore}%`;
    vBarEl.style.width = `${vScore}%`;
    vBarEl.className = `meter-fill ${vScore > 50 ? 'danger' : 'safe'}`;

    // Acoustic Meter
    if (data.acoustic_manipulation_score !== null && data.acoustic_manipulation_score !== undefined) {
      const aScore = Math.round(data.acoustic_manipulation_score * 100);
      aScoreEl.textContent = `${aScore}%`;
      aBarEl.style.width = `${aScore}%`;
      aBarEl.className = `meter-fill ${aScore > 50 ? 'danger' : 'safe'}`;
    } else {
      aScoreEl.textContent = "N/A (Unverified)";
      aBarEl.style.width = "0%";
      aBarEl.className = "meter-fill";
    }

    // Timeline Scrubber
    timelineEl.innerHTML = "";
    if (data.timeline && data.timeline.length > 0) {
      data.timeline.forEach(chunk => {
        const div = document.createElement("div");
        div.className = `timeline-chunk ${chunk.flagged ? 'flagged' : ''}`;
        div.title = `${chunk.timestamp_start}s - ${chunk.timestamp_end}s: ${(chunk.score * 100).toFixed(0)}% manipulation probability`;
        timelineEl.appendChild(div);
      });
    }

    // Speech Captioning & Subtitles (Phase 2A)
    const captionsSection = document.getElementById("media-captions-section");
    const captionTextEl = document.getElementById("media-caption-text");
    const speechStatusEl = document.getElementById("media-speech-status");
    const segsWrapper = document.getElementById("media-caption-segments-wrapper");
    const segsListEl = document.getElementById("media-caption-segments-list");
    const spokenBox = document.getElementById("media-spoken-advisory-box");
    const spokenBadge = document.getElementById("media-spoken-badge");
    const spokenDesc = document.getElementById("media-spoken-desc");
    const spokenIndsList = document.getElementById("media-spoken-indicators-list");

    if (data.transcript) {
      captionsSection.style.display = "block";
      captionTextEl.textContent = `"${data.transcript}"`;

      const isNoSpeech = data.transcript.includes("(No audible speech detected)") || data.transcript.includes("(No speech recognized)");
      if (isNoSpeech) {
        speechStatusEl.className = "caption-status no-speech";
        speechStatusEl.textContent = "No Audible Speech";
      } else {
        speechStatusEl.className = "caption-status";
        speechStatusEl.textContent = "Spoken Dialogue Transcribed";
      }

      // Timestamped Subtitle Chunks
      if (data.transcript_segments && data.transcript_segments.length > 0) {
        segsWrapper.style.display = "block";
        segsListEl.innerHTML = "";
        data.transcript_segments.forEach(seg => {
          const item = document.createElement("div");
          item.className = "segment-chip";
          item.innerHTML = `<span class="segment-time">[${seg.start}s - ${seg.end}s]</span><span class="segment-text">${seg.text}</span>`;
          segsListEl.appendChild(item);
        });
      } else {
        segsWrapper.style.display = "none";
      }

      // Spoken Advisory Compliance
      if (data.spoken_advisory_verdict) {
        spokenBox.style.display = "block";
        spokenBox.className = `spoken-advisory-card ${data.spoken_advisory_verdict}`;
        spokenBadge.textContent = data.spoken_advisory_verdict;
        spokenBadge.className = `verdict-badge ${data.spoken_advisory_verdict === 'CRITICAL_FRAUD' ? 'verdict-critical' : (data.spoken_advisory_verdict === 'SUSPICIOUS' ? 'verdict-warning' : 'verdict-safe')}`;
        
        if (data.spoken_advisory_verdict === "CRITICAL_FRAUD") {
          spokenDesc.textContent = "⚠️ High-risk illegal advisory violations identified in spoken audio (SEBI Investment Advisers Regulations / PFUTP).";
        } else if (data.spoken_advisory_verdict === "SUSPICIOUS") {
          spokenDesc.textContent = "⚠️ Questionable or high-pressure financial claims identified in spoken dialogue.";
        } else {
          spokenDesc.textContent = "✅ Spoken dialogue contains no detected illegal advisory claims or guaranteed returns.";
        }

        spokenIndsList.innerHTML = "";
        (data.spoken_advisory_indicators || []).forEach(ind => {
          const li = document.createElement("li");
          li.textContent = ind;
          spokenIndsList.appendChild(li);
        });
      } else {
        spokenBox.style.display = "none";
      }
    } else {
      captionsSection.style.display = "none";
    }

    // Neural Telemetry & Metadata
    const telemetrySection = document.getElementById("media-neural-telemetry");
    const telemetryDetails = document.getElementById("media-telemetry-details");
    if (data.audio_model_identity) {
      telemetrySection.style.display = "block";
      telemetryDetails.innerHTML = `
        <div class="telemetry-tag">Mode: <strong>${data.audio_inference_mode}</strong></div>
        <div class="telemetry-tag">Audio Model: <strong>${data.audio_model_identity}</strong></div>
        <div class="telemetry-tag">Duration: <strong>${data.audio_duration_seconds ? data.audio_duration_seconds.toFixed(1) + 's' : 'N/A'}</strong></div>
        <div class="telemetry-tag">Sliding Chunks: <strong>${data.audio_chunk_count || 0}</strong></div>
        <div class="telemetry-tag">Peak Chunk Score: <strong>${data.audio_peak_chunk_score !== null && data.audio_peak_chunk_score !== undefined ? (data.audio_peak_chunk_score * 100).toFixed(0) + '%' : 'N/A'}</strong></div>
        <div class="telemetry-tag">Mean Chunk Score: <strong>${data.audio_mean_chunk_score !== null && data.audio_mean_chunk_score !== undefined ? (data.audio_mean_chunk_score * 100).toFixed(0) + '%' : 'N/A'}</strong></div>
      `;
    } else {
      telemetrySection.style.display = "none";
    }

    // Indicators
    indicatorsEl.innerHTML = "";
    (data.key_indicators || []).forEach(ind => {
      const li = document.createElement("li");
      li.textContent = ind;
      indicatorsEl.appendChild(li);
    });

    // Benchmarks
    benchmarksEl.innerHTML = "";
    (data.benchmarks_evaluated || []).forEach(bm => {
      const div = document.createElement("div");
      div.style.marginBottom = "0.5rem";
      div.innerHTML = `<strong>${bm.benchmark_name}</strong>: ${bm.evaluation_metric} — <em>${bm.notes}</em>`;
      benchmarksEl.appendChild(div);
    });
  }
}

// Requirement 2: Impersonation & Chat Forensic Engine
function initMessageScan() {
  const scanBtn = document.getElementById("msg-scan-btn");
  const msgInput = document.getElementById("msg-text-input");
  const platformSelect = document.getElementById("msg-platform-select");
  const resultCard = document.getElementById("msg-result-card");

  // Presets
  document.querySelectorAll(".msg-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.preset;
      if (type === "pump_scam") {
        msgInput.value = "Pakka jackpot rocket call! Guaranteed 100% loss recovery in 2 days. DM to mirror automated trades. Reg: INA000999999 (SEBI approved). Call +91 9876543210 immediately, only 5 slots left!";
      } else if (type === "stolen_reg") {
        msgInput.value = "Join our VIP Telegram group for BankNifty insider options. We are registered Zerodha Broking partners. Use SEBI license INZ000031633. 50% guaranteed monthly profit.";
      } else {
        msgInput.value = "Disclaimer: Equity investments are subject to market risks. Please read all scheme-related documents carefully before investing. Consult a SEBI certified RIA.";
      }
    });
  });

  if (scanBtn) {
    scanBtn.addEventListener("click", async () => {
      const text = msgInput.value.trim();
      if (!text) return alert("Please enter message text to scan.");

      setLoading(scanBtn, true);
      try {
        const res = await fetch("/api/scan/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message_text: text,
            platform: platformSelect.value
          })
        });
        const data = await res.json();
        renderMessageResult(data);
      } catch (err) {
        alert("Scan failed: " + err.message);
      } finally {
        setLoading(scanBtn, false);
      }
    });
  }

  function renderMessageResult(data) {
    resultCard.style.display = "block";
    const badge = document.getElementById("msg-verdict-badge");
    const riskScoreEl = document.getElementById("msg-risk-score");
    const riskBarEl = document.getElementById("msg-risk-bar");
    const regBadge = document.getElementById("msg-reg-status-badge");
    const sanitizedEl = document.getElementById("msg-sanitized-text");
    const piiEl = document.getElementById("msg-pii-redacted");
    const hinglishEl = document.getElementById("msg-hinglish-tags");
    const violationsEl = document.getElementById("msg-violations-list");

    if (data.is_suspicious_advisory) {
      badge.className = "verdict-badge verdict-critical";
      badge.textContent = "SUSPICIOUS / ILLEGAL ADVISORY";
    } else {
      badge.className = "verdict-badge verdict-safe";
      badge.textContent = "SAFE ADVISORY CONTENT";
    }

    const rScore = Math.round(data.composite_risk_score * 100);
    riskScoreEl.textContent = `${rScore}%`;
    riskBarEl.style.width = `${rScore}%`;
    riskBarEl.className = `meter-fill ${rScore >= 60 ? 'danger' : rScore >= 35 ? 'warn' : 'safe'}`;

    // SEBI Registry Status Badge
    const st = data.advisor_claim ? data.advisor_claim.registry_status : "NOT_CLAIMED";
    regBadge.textContent = `SEBI Status: ${st}`;
    if (st === "VERIFIED") {
      regBadge.className = "pill";
      regBadge.style.color = "var(--accent-emerald)";
    } else if (st === "MISMATCH" || st === "UNREGISTERED") {
      regBadge.className = "pill";
      regBadge.style.color = "var(--accent-rose)";
    } else {
      regBadge.className = "pill";
      regBadge.style.color = "var(--text-muted)";
    }

    sanitizedEl.textContent = data.sanitized_text;

    // Redacted PII
    piiEl.innerHTML = "";
    if (data.redacted_pii && data.redacted_pii.length > 0) {
      data.redacted_pii.forEach(p => {
        const span = document.createElement("span");
        span.className = "pill";
        span.textContent = `🛡️ Redacted: ${p}`;
        piiEl.appendChild(span);
      });
    } else {
      piiEl.textContent = "No sensitive PII detected.";
    }

    // Hinglish Slang
    hinglishEl.innerHTML = "";
    if (data.detected_hinglish_slang && data.detected_hinglish_slang.length > 0) {
      data.detected_hinglish_slang.forEach(s => {
        const span = document.createElement("span");
        span.className = "pill";
        span.style.color = "var(--accent-amber)";
        span.textContent = `⚠️ Colloquialism: "${s}"`;
        hinglishEl.appendChild(span);
      });
    }

    // Violations
    violationsEl.innerHTML = "";
    (data.violations || []).forEach(v => {
      const li = document.createElement("li");
      li.textContent = v;
      violationsEl.appendChild(li);
    });
  }
}

// Requirement 3: App & Phishing Infrastructure Engine
function initAppScan() {
  const scanBtn = document.getElementById("app-scan-btn");
  const urlInput = document.getElementById("app-domain-input");
  const pkgInput = document.getElementById("app-pkg-input");
  const permsInput = document.getElementById("app-perms-input");
  const resultCard = document.getElementById("app-result-card");

  // Presets
  document.querySelectorAll(".app-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.preset;
      if (type === "fake_groww") {
        urlInput.value = "https://groww-bonus-reward.xyz/login";
        pkgInput.value = "com.groww.vip.trade";
        permsInput.value = "READ_SMS, RECEIVE_SMS, BIND_ACCESSIBILITY_SERVICE, SYSTEM_ALERT_WINDOW";
      } else if (type === "fake_zerodha") {
        urlInput.value = "https://zerodha-trading-terminal.top";
        pkgInput.value = "com.zerodha.quicktrade";
        permsInput.value = "READ_SMS, PACKAGE_USAGE_STATS";
      } else {
        urlInput.value = "https://zerodha.com";
        pkgInput.value = "com.zerodha.kite3";
        permsInput.value = "INTERNET, ACCESS_NETWORK_STATE";
      }
    });
  });

  if (scanBtn) {
    scanBtn.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      const pkg = pkgInput.value.trim();
      const permsRaw = permsInput.value.trim();
      const perms = permsRaw ? permsRaw.split(",").map(p => p.trim()) : [];

      if (!url && !pkg && perms.length === 0) {
        return alert("Please enter a domain URL, package name, or permissions list.");
      }

      setLoading(scanBtn, true);
      try {
        const res = await fetch("/api/scan/app", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            target_url: url || null,
            package_name: pkg || null,
            declared_permissions: perms
          })
        });
        const data = await res.json();
        renderAppResult(data);
      } catch (err) {
        alert("Scan failed: " + err.message);
      } finally {
        setLoading(scanBtn, false);
      }
    });
  }

  function renderAppResult(data) {
    resultCard.style.display = "block";
    const badge = document.getElementById("app-verdict-badge");
    const riskScoreEl = document.getElementById("app-risk-score");
    const riskBarEl = document.getElementById("app-risk-bar");
    const wlStatus = document.getElementById("app-whitelist-badge");
    const domainDetails = document.getElementById("app-domain-details");
    const permsList = document.getElementById("app-perms-list");

    if (data.is_fraudulent) {
      badge.className = "verdict-badge verdict-critical";
      badge.textContent = "FRAUDULENT TRADING APP / DOMAIN";
    } else {
      badge.className = "verdict-badge verdict-safe";
      badge.textContent = "LEGITIMATE / WHITELISTED";
    }

    const rScore = Math.round(data.risk_score * 100);
    riskScoreEl.textContent = `${rScore}%`;
    riskBarEl.style.width = `${rScore}%`;
    riskBarEl.className = `meter-fill ${rScore >= 60 ? 'danger' : 'safe'}`;

    wlStatus.textContent = `Status: ${data.whitelist_status}`;
    if (data.whitelist_status === "OFFICIAL_WHITELISTED") {
      wlStatus.style.color = "var(--accent-emerald)";
    } else {
      wlStatus.style.color = "var(--accent-rose)";
    }

    // Domain Details
    if (data.domain_result) {
      const d = data.domain_result;
      domainDetails.innerHTML = `
        <p><strong>Input Domain:</strong> ${d.input_domain}</p>
        <p><strong>Impersonation Target:</strong> ${d.impersonated_broker || 'None'}</p>
        <p><strong>Levenshtein Distance:</strong> ${d.levenshtein_distance >= 0 ? d.levenshtein_distance : 'N/A'}</p>
        <p><strong>Evaluation Reasons:</strong> ${d.reasons.join('; ')}</p>
      `;
    } else {
      domainDetails.innerHTML = "<p>No domain specified.</p>";
    }

    // Permissions List
    permsList.innerHTML = "";
    if (data.apk_result && data.apk_result.high_risk_permissions.length > 0) {
      data.apk_result.high_risk_permissions.forEach(p => {
        const li = document.createElement("li");
        li.style.color = "var(--accent-rose)";
        li.textContent = `🚨 ${p}`;
        permsList.appendChild(li);
      });
    } else {
      permsList.innerHTML = "<li>No high-risk surveillance permissions declared.</li>";
    }
  }
}

// Live Alerts Feed
async function initAlertsFeed() {
  const container = document.getElementById("alerts-feed-container");
  const refreshBtn = document.getElementById("refresh-alerts-btn");

  async function fetchAlerts() {
    try {
      const res = await fetch("/api/alerts/recent?limit=15");
      const alerts = await res.json();
      renderAlerts(alerts);
    } catch (err) {
      console.warn("Alerts fetch error:", err);
    }
  }

  function renderAlerts(alerts) {
    if (!container) return;
    container.innerHTML = "";
    if (!alerts || alerts.length === 0) {
      container.innerHTML = "<p style='color: var(--text-muted); padding: 1rem;'>No active threats detected yet. All vectors nominal.</p>";
      return;
    }

    alerts.slice().reverse().forEach(a => {
      const div = document.createElement("div");
      div.className = `alert-item ${a.severity}`;
      div.innerHTML = `
        <div class="alert-meta">
          <span>${a.category} • Risk: ${(a.risk_score * 100).toFixed(0)}%</span>
          <span>${new Date(a.timestamp).toLocaleTimeString()}</span>
        </div>
        <strong>${a.summary}</strong>
        <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">${a.action_required}</p>
      `;
      container.appendChild(div);
    });
  }

  if (refreshBtn) refreshBtn.addEventListener("click", fetchAlerts);
  fetchAlerts();
  // Auto poll every 10s
  setInterval(fetchAlerts, 10000);
}

// SEBI SCORES Dossier Generator
function initDossierGenerator() {
  const generateBtn = document.getElementById("dossier-generate-btn");
  const copyBtn = document.getElementById("dossier-copy-btn");
  const dossierView = document.getElementById("dossier-text-view");

  if (generateBtn) {
    generateBtn.addEventListener("click", async () => {
      setLoading(generateBtn, true);
      try {
        const res = await fetch("/api/report/generate-dossier", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message_text: "Guaranteed 100% loss recovery call! DM to mirror trades. SEBI: INA999999",
            target_url: "https://groww-bonus-reward.xyz",
            package_name: "com.groww.vip.trade",
            app_label: "Groww VIP Reward App"
          })
        });
        const data = await res.json();
        const formatted = `
================================================================================
           SECURITIES AND EXCHANGE BOARD OF INDIA (SEBI)
       COMPLAINTS REDRESS SYSTEM (SCORES) EVIDENCE DOSSIER
================================================================================
Dossier Reference ID : ${data.dossier_id}
Filing Timestamp     : ${data.generated_at}
Alleged Entity/Handle: ${data.alleged_entity_or_handle}

STATUTORY PROVISIONS CITED:
${data.regulatory_acts_cited.map(a => ` - ${a}`).join('\n')}

FACTUAL SUMMARY OF INVESTIGATION:
${data.factual_summary}

EVIDENTIARY AUDIT LOG:
${data.evidence_points.map(e => ` [EVIDENCE] ${e}`).join('\n')}

RECOMMENDED REGULATORY ACTION:
${data.recommended_sebi_action}
================================================================================
Generated by FinGuard 2.0 Autonomous Forensic Security System.
        `.trim();

        dossierView.textContent = formatted;
        if (copyBtn) copyBtn.style.display = "inline-flex";
      } catch (err) {
        alert("Dossier generation failed: " + err.message);
      } finally {
        setLoading(generateBtn, false);
      }
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(dossierView.textContent);
      copyBtn.textContent = "Copied to Clipboard!";
      setTimeout(() => copyBtn.textContent = "Copy Dossier Text", 2000);
    });
  }
}

function setLoading(btn, isLoading) {
  if (!btn) return;
  if (isLoading) {
    btn.dataset.originalText = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span> Processing...`;
    btn.disabled = true;
  } else {
    btn.innerHTML = btn.dataset.originalText || "Scan";
    btn.disabled = false;
  }
}
