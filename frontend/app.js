const CHAT_ENDPOINT = "/chat";

const TRUST_LABEL_CLASS = {
  "Reliable": "high",
  "Needs checking": "mid",
  "Unreliable": "low",
};

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question-input");
const sendButtonEl = document.getElementById("send-button");
const qualityPresetEl = document.getElementById("quality-preset");

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addBotMessage(data) {
  const row = document.createElement("div");
  row.className = "message bot";

  const textEl = document.createElement("div");
  textEl.textContent = data.response;
  row.appendChild(textEl);

  // tlm only says something substantive below its own 0.8 threshold, and only
  // when reasoning_effort is not "none". Above that it is a fixed string, so
  // showing it would just add noise to every answer.
  if (data.explanation && data.trust_score < 0.8) {
    const why = document.createElement("details");
    why.className = "explanation";
    const summary = document.createElement("summary");
    summary.textContent = "Why this score?";
    why.appendChild(summary);
    const body = document.createElement("div");
    body.textContent = data.explanation;
    why.appendChild(body);
    row.appendChild(why);
  }

  const meta = document.createElement("div");
  meta.className = "bot-meta";
  const trustScore = data.trust_score;
  const trustLabel = data.trust_label;
  const durationS = data.duration_s;
  const qualityPreset = data.quality_preset;

  const badge = document.createElement("span");
  const badgeClass = TRUST_LABEL_CLASS[trustLabel] || "mid";
  badge.className = `trust-badge ${badgeClass}`;
  badge.title = `Raw trust score: ${trustScore.toFixed(3)}`;
  // Built from nodes rather than innerHTML: the label is server-controlled today,
  // but this is the only unescaped sink in the file and the cost is the same.
  badge.appendChild(document.createTextNode(`${trustLabel} `));
  const scoreEl = document.createElement("span");
  scoreEl.className = "trust-score";
  scoreEl.textContent = `(${trustScore.toFixed(2)})`;
  badge.appendChild(scoreEl);
  meta.appendChild(badge);

  const duration = document.createElement("span");
  duration.className = "duration-tag";
  const details = [`⏱ ${durationS.toFixed(1)}s`, qualityPreset];
  if (data.perplexity != null) details.push(`perplexity ${data.perplexity.toFixed(2)}`);
  if (data.completion_tokens != null) details.push(`${data.completion_tokens} tok`);
  duration.textContent = details.join(" · ");
  duration.title = [data.generator_model, data.judge_model].filter(Boolean).join("  judged by  ");
  meta.appendChild(duration);

  row.appendChild(meta);

  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;

  addMessage("user", question);
  inputEl.value = "";
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  const qualityPreset = qualityPresetEl.value;

  const pendingStart = performance.now();
  const pendingEl = addMessage("pending", "");
  const tick = () => {
    const elapsed = (performance.now() - pendingStart) / 1000;
    pendingEl.textContent =
      `Generating the answer, then computing the trust score (${qualityPreset})... (${elapsed.toFixed(0)}s elapsed)`;
  };
  tick();
  const tickInterval = setInterval(tick, 1000);

  try {
    const res = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, quality_preset: qualityPreset }),
    });

    const data = await res.json();
    clearInterval(tickInterval);
    pendingEl.remove();

    if (!res.ok) {
      addMessage("error", data.detail || "Something went wrong.");
    } else {
      addBotMessage(data);
    }
  } catch (err) {
    clearInterval(tickInterval);
    pendingEl.remove();
    addMessage("error", "Could not reach the server. Is it running on port 8000?");
  } finally {
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  }
});
