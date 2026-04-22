async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "request failed");
  }
  return response.json();
}

function metricCard(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderOverview(data) {
  const availableModels = data.model_entries
    .map(item => `${item.name}${item.available ? "" : " (未就绪)"}`)
    .join("<br>");
  document.getElementById("overviewCard").innerHTML = `
    <div class="stat"><span>模态</span><strong>${data.modality}</strong></div>
    <div class="stat"><span>任务</span><strong>${data.tasks.join(", ")}</strong></div>
    <div class="stat"><span>模型</span><strong>${availableModels}</strong></div>
    <div class="stat"><span>实验数</span><strong>${data.experiment_count}</strong></div>
  `;

  document.getElementById("datasetSelect").innerHTML = data.datasets
    .map(item => `<option value="${item.dataset_id}">${item.name}</option>`)
    .join("");

  document.getElementById("modelSelect").innerHTML = data.model_entries
    .map(item => `<option value="${item.name}" ${item.available ? "" : "disabled"}>${item.name}${item.available ? "" : " (未就绪)"}</option>`)
    .join("");

  document.getElementById("clipHint").textContent = data.clip_status.available
    ? `CLIP backend: ${data.clip_status.backend}`
    : data.clip_status.message;
}

function renderSamples(payload) {
  document.getElementById("datasetMeta").textContent = `样本总数 ${payload.total}`;
  document.getElementById("sampleGallery").innerHTML = payload.samples.map(sample => `
    <article class="sample-card">
      <img src="/${sample.path}" alt="${sample.sample_id}" />
      <div>
        <strong>${sample.sample_id}</strong>
        <p>${sample.label}</p>
        <p>${sample.caption}</p>
      </div>
    </article>
  `).join("");
}

function renderMetrics(result) {
  const m = result.aggregate_metrics;
  document.getElementById("metricsCards").innerHTML = [
    metricCard("攻击成功率", `${(m.attack_success_rate * 100).toFixed(1)}%`),
    metricCard("迁移成功率", m.transfer_success_rate === null ? "N/A" : `${(m.transfer_success_rate * 100).toFixed(1)}%`),
    metricCard("平均 Linf", m.avg_linf),
    metricCard("平均 L2", m.avg_l2),
    metricCard("平均偏移", m.avg_confidence_shift),
    metricCard("语义一致性", m.avg_semantic_consistency),
    metricCard("平均查询次数", m.avg_queries),
  ].join("");

  const entries = [
    ["ASR", m.attack_success_rate * 100],
    ["Transfer", (m.transfer_success_rate || 0) * 100],
    ["Linf", Math.min(m.avg_linf * 100, 100)],
    ["Shift", Math.max(m.avg_confidence_shift * 100, 0)],
    ["Semantic", Math.min(m.avg_semantic_consistency * 100, 100)],
  ];
  document.getElementById("metricsChart").innerHTML = `
    <div class="bar-chart">
      ${entries.map(([label, value]) => `
        <div class="bar-row">
          <span>${label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.min(value, 100)}%"></div></div>
          <strong>${value.toFixed(1)}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function renderComparisons(result) {
  document.getElementById("comparisonList").innerHTML = result.records.map(item => `
    <article class="compare-card">
      <div class="compare-images">
        <img src="/${item.original_image}" alt="original" />
        <img src="/${item.adversarial_image}" alt="adversarial" />
      </div>
      <div class="compare-meta">
        <strong>${item.sample_id}</strong>
        <p>真实标签: ${item.label}</p>
        <p>原始预测: ${item.original_prediction}</p>
        <p>攻击后预测: ${item.adversarial_prediction}</p>
        <p>目标标签: ${item.target_label || "N/A"}</p>
        <p>问题: ${item.question || "N/A"}</p>
        <p>提示模板: ${item.prompt_template || "N/A"}</p>
        <p>攻击细节: ${JSON.stringify(item.attack_debug || {})}</p>
        <p>成功: ${item.metrics.success ? "是" : "否"} / Linf: ${item.metrics.linf} / 查询: ${item.metrics.queries}</p>
      </div>
    </article>
  `).join("");
}

function renderHistory(items) {
  document.getElementById("experimentHistory").innerHTML = items.map(item => `
    <article class="history-item">
      <strong>${item.experiment_id}</strong>
      <p>${item.created_at}</p>
      <p>${item.model_name} / ${item.attack_name}</p>
      <p>ASR ${(item.aggregate_metrics.attack_success_rate * 100).toFixed(1)}%</p>
    </article>
  `).join("");
}

async function loadSamples() {
  const datasetId = document.getElementById("datasetSelect").value;
  const payload = await fetchJson(`/api/datasets/${datasetId}/samples?limit=12`);
  renderSamples(payload);
}

async function loadHistory() {
  const payload = await fetchJson("/api/experiments");
  renderHistory(payload);
}

function syncModelState() {
  const model = document.getElementById("modelSelect").value;
  const attackSelect = document.getElementById("attackSelect");
  const scenarioSelect = document.getElementById("scenarioSelect");
  const isClip = model.startsWith("clip_");
  const isApi = model.startsWith("api_");
  const isSurrogate = !isClip && !isApi;

  Array.from(attackSelect.options).forEach(option => {
    option.disabled = false;
    if (isClip && option.value === "prompt_injection") {
      option.disabled = false;
    }
    if (isApi && (option.value === "fgsm" || option.value === "pgd")) {
      option.disabled = true;
    }
    if (isSurrogate && option.value === "prompt_injection") {
      option.disabled = true;
    }
  });

  if (isClip) scenarioSelect.value = "clip_zero_shot_classification";
  if (isApi) scenarioSelect.value = attackSelect.value === "prompt_injection" ? "api_prompt_injection" : "api_blackbox_attack";
  if (isSurrogate) scenarioSelect.value = "surrogate_transfer_attack";

  document.getElementById("promptTemplateInput").disabled = !isClip;
  document.getElementById("injectionPromptInput").disabled = !(isClip || isApi);
  document.getElementById("apiBaseUrlInput").disabled = !isApi || model === "api_mock_vision";
  document.getElementById("apiKeyInput").disabled = !isApi || model === "api_mock_vision";
  document.getElementById("apiModelInput").disabled = !isApi || model === "api_mock_vision";
}

async function runAttack() {
  const payload = {
    dataset_id: document.getElementById("datasetSelect").value,
    model_name: document.getElementById("modelSelect").value,
    scenario: document.getElementById("scenarioSelect").value,
    attack_name: document.getElementById("attackSelect").value,
    epsilon: Number(document.getElementById("epsilonInput").value),
    alpha: Number(document.getElementById("alphaInput").value),
    steps: Number(document.getElementById("stepsInput").value),
    max_samples: Number(document.getElementById("sampleCountInput").value),
    targeted: document.getElementById("targetedCheckbox").checked,
    target_label: document.getElementById("targetLabelSelect").value || null,
    prompt_template: document.getElementById("promptTemplateInput").value,
    injection_prompt: document.getElementById("injectionPromptInput").value,
    api_base_url: document.getElementById("apiBaseUrlInput").value || null,
    api_key: document.getElementById("apiKeyInput").value || null,
    api_model: document.getElementById("apiModelInput").value || null,
  };

  const button = document.getElementById("runBtn");
  button.disabled = true;
  button.textContent = "运行中...";
  try {
    const result = await fetchJson("/api/attacks/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderMetrics(result);
    renderComparisons(result);
    await loadHistory();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "运行实验";
  }
}

async function init() {
  const overview = await fetchJson("/api/overview");
  const runtimeConfig = await fetchJson("/api/runtime-config");
  renderOverview(overview);
  const apiDefaults = runtimeConfig.api_defaults || {};
  document.getElementById("apiBaseUrlInput").value = apiDefaults.base_url || "";
  document.getElementById("apiModelInput").value = apiDefaults.model || "";
  if (apiDefaults.api_key_configured) {
    document.getElementById("apiKeyInput").placeholder = `已配置: ${apiDefaults.api_key_preview}`;
  }
  syncModelState();
  await loadSamples();
  await loadHistory();
}

document.getElementById("datasetSelect").addEventListener("change", loadSamples);
document.getElementById("modelSelect").addEventListener("change", syncModelState);
document.getElementById("attackSelect").addEventListener("change", syncModelState);
document.getElementById("runBtn").addEventListener("click", runAttack);
init();
