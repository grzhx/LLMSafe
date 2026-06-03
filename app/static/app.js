async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "request failed");
  }
  return response.json();
}

let scenarioCatalog = [];
let modelEntries = [];
let currentRunId = null;
let currentRunTimer = null;
let currentSelectedExperimentId = null;
let comparisonExpanded = false;
let comparisonFilter = "all";
let currentExperimentDetail = null;
const collapsedComparisonCount = 1;

function metricCard(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
}

function setProgress(state) {
  const progress = Math.max(0, Math.min(100, Math.round((state.progress || 0) * 100)));
  document.getElementById("runStatusText").textContent = state.message || "等待中";
  document.getElementById("runProgressLabel").textContent = `${progress}%`;
  document.getElementById("runProgressDetail").textContent = `${state.completed_samples || 0} / ${state.total_samples || 0}`;
  document.getElementById("runProgressBar").style.width = `${progress}%`;
}

function resetProgress() {
  setProgress({
    message: "当前没有正在运行的实验。",
    progress: 0,
    completed_samples: 0,
    total_samples: 0,
  });
}

function renderOverview(data) {
  modelEntries = data.model_entries || [];
  const modelSummary = modelEntries
    .map(item => `${item.name}${item.available ? "" : "（不可用）"}`)
    .join("<br>");
  document.getElementById("overviewCard").innerHTML = `
    <div class="stat"><span>模态</span><strong>${data.modality}</strong></div>
    <div class="stat"><span>覆盖任务</span><strong>${data.tasks.length}</strong></div>
    <div class="stat"><span>模型能力</span><strong>${modelSummary}</strong></div>
    <div class="stat"><span>实验记录</span><strong>${data.experiment_count}</strong></div>
  `;
  document.getElementById("clipHint").textContent = data.clip_status.available
    ? `CLIP 后端已可用：${data.clip_status.backend}`
    : data.clip_status.message;
}

function fillSelect(id, items, valueKey, labelKey) {
  document.getElementById(id).innerHTML = items
    .map(item => `<option value="${item[valueKey]}">${item[labelKey]}</option>`)
    .join("");
}

function selectedScenario() {
  const scenarioId = document.getElementById("scenarioSelect").value;
  return scenarioCatalog.find(item => item.scenario_id === scenarioId);
}

function renderScenarioCard() {
  const scenario = selectedScenario();
  if (!scenario) return;
  document.getElementById("scenarioCard").innerHTML = `
    <strong>${scenario.name}</strong>
    <p>${scenario.description}</p>
    <p>任务类型：${scenario.task_type}</p>
    <p>支持模型：${scenario.supported_models.join(", ")}</p>
    <p>支持攻击：${scenario.supported_attacks.join(", ")}</p>
  `;
}

function renderSamples(payload) {
  document.getElementById("datasetMeta").textContent = `当前展示 ${payload.samples.length} / 总计 ${payload.total}`;
  document.getElementById("sampleGallery").innerHTML = payload.samples.map(sample => `
    <article class="sample-card">
      <img src="/${sample.path}" alt="${sample.sample_id}" />
      <div>
        <strong>${sample.sample_id}</strong>
        <p>标签：${sample.label || "N/A"}</p>
        <p>${sample.question || sample.caption || "无附加文本"}</p>
      </div>
    </article>
  `).join("");
}

function renderDatasetSummary(summary) {
  const labels = Object.entries(summary.label_counts || {})
    .slice(0, 12)
    .map(([label, count]) => `${label}: ${count}`)
    .join(" / ");
  document.getElementById("datasetSummary").innerHTML = `
    <strong>${summary.meta.name}</strong>
    <p>${summary.meta.description || ""}</p>
    <p>版本：${summary.meta.version || "N/A"}</p>
    <p>任务：${summary.meta.task || "N/A"}</p>
    <p>样本总数：${summary.total}</p>
    <p>标签统计：${labels || "无"}</p>
    <p>样本版本：${(summary.versions || []).join(", ")}</p>
  `;
}

function renderMetrics(result) {
  const m = result.aggregate_metrics || {};
  const cards = [
    metricCard("攻击成功率", `${((m.attack_success_rate || 0) * 100).toFixed(1)}%`),
    metricCard("平均 Linf", m.avg_linf ?? "N/A"),
    metricCard("平均 L2", m.avg_l2 ?? "N/A"),
    metricCard("平均偏移", m.avg_confidence_shift ?? "N/A"),
    metricCard("语义一致性", m.avg_semantic_consistency ?? "N/A"),
    metricCard("平均查询数", m.avg_queries ?? "N/A"),
  ];
  if (m.retrieval_recall_at_1 !== undefined) {
    cards.push(metricCard("Recall@1", `${(m.retrieval_recall_at_1 * 100).toFixed(1)}%`));
  }
  if (m.retrieval_recall_at_3 !== undefined) {
    cards.push(metricCard("Recall@3", `${(m.retrieval_recall_at_3 * 100).toFixed(1)}%`));
  }
  if (m.mean_rank_shift !== undefined) {
    cards.push(metricCard("平均排名变化", m.mean_rank_shift));
  }
  if (m.mean_average_precision_proxy !== undefined) {
    cards.push(metricCard("mAP 代理", m.mean_average_precision_proxy));
  }
  if (m.answer_shift_rate !== undefined) {
    cards.push(metricCard("回答偏移率", `${(m.answer_shift_rate * 100).toFixed(1)}%`));
  }
  if (m.constraint_violation_rate !== undefined) {
    cards.push(metricCard("约束违背率", `${(m.constraint_violation_rate * 100).toFixed(1)}%`));
  }
  if (m.goal_hijack_rate !== undefined) {
    cards.push(metricCard("任务劫持率", `${(m.goal_hijack_rate * 100).toFixed(1)}%`));
  }
  document.getElementById("metricsCards").innerHTML = cards.join("");

  const entries = [
    ["ASR", (m.attack_success_rate || 0) * 100],
    ["Linf", Math.min((m.avg_linf || 0) * 100, 100)],
    ["Semantic", Math.min((m.avg_semantic_consistency || 0) * 100, 100)],
    ["Queries", Math.min((m.avg_queries || 0) * 5, 100)],
  ];
  if (m.retrieval_recall_at_1 !== undefined) {
    entries.push(["R@1", m.retrieval_recall_at_1 * 100]);
  }
  if (m.answer_shift_rate !== undefined) {
    entries.push(["Shift", m.answer_shift_rate * 100]);
  }
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
  const records = result.records || [];
  const filteredRecords = filterComparisonRecords(records);
  const toggleButton = document.getElementById("toggleComparisonBtn");
  const filterCount = document.getElementById("comparisonFilterCount");
  if (filterCount) {
    filterCount.textContent = `${filteredRecords.length} / ${records.length}`;
  }
  if (!records.length) {
    document.getElementById("comparisonList").innerHTML = "<p>暂无对比数据。</p>";
    toggleButton.style.display = "none";
    return;
  }
  if (!filteredRecords.length) {
    document.getElementById("comparisonList").innerHTML = "<p>当前筛选条件下没有样本。</p>";
    toggleButton.style.display = "none";
    return;
  }
  const visibleRecords = comparisonExpanded ? filteredRecords : filteredRecords.slice(0, collapsedComparisonCount);
  document.getElementById("comparisonList").innerHTML = visibleRecords.map(item => `
    <article class="compare-card">
      <div class="compare-images">
        <img src="/${item.original_image}" alt="original" />
        <img src="/${item.adversarial_image}" alt="adversarial" />
      </div>
      <div class="compare-meta">
        <strong>${item.sample_id}</strong>
        <p>场景：${item.scenario || result.scenario}</p>
        <p>原始输出：${item.original_prediction}</p>
        <p>攻击后输出：${item.adversarial_prediction}</p>
        <p>成功：${item.metrics?.success ? "是" : "否"} / Linf：${item.metrics?.linf ?? "N/A"} / 查询：${item.metrics?.queries ?? "N/A"}</p>
        <details class="compare-details">
          <summary>展开详情</summary>
          <div class="compare-extra">
            <p>目标文本：${item.target_label || "自动"}</p>
            <p>问题：${item.question || "N/A"}</p>
            <p>注入来源：${item.attack_source || "N/A"} / 强度：${item.injection_strength || "N/A"}</p>
            <p>约束违背：${item.constraint_violated ? "是" : "否"}</p>
            <p>排名变化：${item.rank_shift ?? "N/A"} / Recall@1：${item.retrieval_recall_at_1 ?? "N/A"}</p>
            <p>候选项：${(item.candidate_texts || []).slice(0, 5).join(" | ") || "N/A"}</p>
            <pre>${JSON.stringify(item.attack_debug || {}, null, 2)}</pre>
          </div>
        </details>
      </div>
    </article>
  `).join("");
  if (filteredRecords.length <= collapsedComparisonCount) {
    toggleButton.style.display = "none";
  } else {
    toggleButton.style.display = "inline-flex";
    toggleButton.textContent = comparisonExpanded
      ? `收起（当前显示 ${visibleRecords.length} / ${filteredRecords.length}）`
      : `展开全部（当前显示 ${visibleRecords.length} / ${filteredRecords.length}）`;
  }
}

function filterComparisonRecords(records) {
  if (comparisonFilter === "success") {
    return records.filter(item => Boolean(item.metrics?.success));
  }
  if (comparisonFilter === "failed") {
    return records.filter(item => !item.metrics?.success);
  }
  if (comparisonFilter === "shifted") {
    return records.filter(item => item.answer_shifted || item.original_prediction !== item.adversarial_prediction);
  }
  if (comparisonFilter === "unchanged") {
    return records.filter(item => !(item.answer_shifted || item.original_prediction !== item.adversarial_prediction));
  }
  return records;
}

function renderExperimentDetail(result) {
  currentExperimentDetail = result;
  const params = result.parameters || {};
  const aggregate = result.aggregate_metrics || {};
  document.getElementById("experimentDetail").innerHTML = `
    <strong>${result.experiment_id}</strong>
    <p>创建时间：${result.created_at}</p>
    <p>数据集：${result.dataset_id}</p>
    <p>模型：${result.model_name}</p>
    <p>场景：${result.scenario}</p>
    <p>攻击：${result.attack_name}</p>
    <p>样本数：${(result.records || []).length}</p>
    <p>关键指标：ASR=${aggregate.attack_success_rate ?? "N/A"}，Linf=${aggregate.avg_linf ?? "N/A"}，L2=${aggregate.avg_l2 ?? "N/A"}，Queries=${aggregate.avg_queries ?? "N/A"}</p>
    <details>
      <summary>参数配置</summary>
      <pre>${JSON.stringify(params, null, 2)}</pre>
    </details>
    <details>
      <summary>聚合指标</summary>
      <pre>${JSON.stringify(aggregate, null, 2)}</pre>
    </details>
  `;
  renderMetrics(result);
  renderComparisons(result);
}

function markSelectedExperiment(experimentId) {
  currentSelectedExperimentId = experimentId;
  document.querySelectorAll(".history-clickable").forEach(node => {
    node.classList.toggle("selected", node.dataset.id === experimentId);
  });
}

async function openExperimentDetail(experimentId) {
  try {
    const detail = await fetchJson(`/api/experiments/${experimentId}`);
    markSelectedExperiment(experimentId);
    comparisonExpanded = false;
    renderExperimentDetail(detail);
  } catch (error) {
    alert(`加载实验详情失败：${error.message}`);
  }
}

window.openExperimentDetailById = openExperimentDetail;

function renderHistory(items) {
  document.getElementById("experimentHistory").innerHTML = items.map(item => `
    <article class="history-item history-clickable" data-id="${item.experiment_id}" role="button" tabindex="0" onclick="window.openExperimentDetailById('${item.experiment_id}')">
      <strong>${item.experiment_id}</strong>
      <p>${item.created_at}</p>
      <p>${item.scenario} / ${item.attack_name}</p>
      <p>${item.model_name}</p>
      <p>ASR ${((item.aggregate_metrics?.attack_success_rate || 0) * 100).toFixed(1)}%</p>
    </article>
  `).join("");
  markSelectedExperiment(currentSelectedExperimentId);
}

function renderStatsBoard(stats) {
  const attackStats = Object.entries(stats.attack_stats || {})
    .map(([k, v]) => `<p>${k}: ${(v * 100).toFixed(1)}%</p>`).join("");
  const modelStats = Object.entries(stats.model_stats || {})
    .map(([k, v]) => `<p>${k}: ${(v * 100).toFixed(1)}%</p>`).join("");
  document.getElementById("statsBoard").innerHTML = `
    <div class="stats-grid">
      <div class="info-card">
        <strong>攻击方法平均成功率</strong>
        ${attackStats || "<p>暂无数据</p>"}
      </div>
      <div class="info-card">
        <strong>模型平均受攻击成功率</strong>
        ${modelStats || "<p>暂无数据</p>"}
      </div>
    </div>
  `;
}

function renderCompareBoard(payload) {
  document.getElementById("compareBoard").innerHTML = payload.summary.map(item => `
    <article class="history-item">
      <strong>${item.experiment_id}</strong>
      <p>场景：${item.scenario}</p>
      <p>模型：${item.model_name}</p>
      <p>攻击：${item.attack_name}</p>
      <p>ASR：${((item.attack_success_rate || 0) * 100).toFixed(1)}%</p>
      <p>平均查询：${item.avg_queries}</p>
      <p>平均 Linf：${item.avg_linf}</p>
    </article>
  `).join("");
}

async function loadSamplesAndSummary() {
  const datasetId = document.getElementById("datasetSelect").value;
  if (!datasetId) return;
  const [samples, summary] = await Promise.all([
    fetchJson(`/api/datasets/${datasetId}/samples?limit=12`),
    fetchJson(`/api/datasets/${datasetId}/summary`),
  ]);
  renderSamples(samples);
  renderDatasetSummary(summary);
}

async function loadHistoryAndStats() {
  const [history, stats] = await Promise.all([
    fetchJson("/api/experiments"),
    fetchJson("/api/experiments/stats"),
  ]);
  renderHistory(history);
  renderStatsBoard(stats);
  if (!currentSelectedExperimentId && history.length > 0) {
    await openExperimentDetail(history[0].experiment_id);
  }
}

function renderModelOptions() {
  const scenario = selectedScenario();
  const models = modelEntries.filter(item => {
    if (!item.available) return false;
    if (!scenario) return true;
    return scenario.supported_models.includes(item.name);
  });
  const select = document.getElementById("modelSelect");
  const current = select.value;
  select.innerHTML = models.map(item => `<option value="${item.name}">${item.name}</option>`).join("");
  if (models.some(item => item.name === current)) {
    select.value = current;
  }
}

function syncImportLabels() {
  const type = document.getElementById("importTypeSelect").value;
  const label1 = document.getElementById("importJsonLabel1");
  const label2 = document.getElementById("importJsonLabel2");
  if (type === "vqav2") {
    label1.firstChild.textContent = "questions JSON";
    label2.firstChild.textContent = "annotations JSON";
    document.getElementById("importJsonInput2").disabled = false;
  } else {
    label1.firstChild.textContent = "captions JSON";
    label2.firstChild.textContent = "annotations JSON（COCO 可留空）";
    document.getElementById("importJsonInput2").disabled = true;
    document.getElementById("importJsonInput2").value = "";
  }
}

function syncFormState() {
  renderModelOptions();
  const model = document.getElementById("modelSelect").value;
  const attackSelect = document.getElementById("attackSelect");
  const scenario = selectedScenario();
  const isClip = model.startsWith("clip_");
  const isApi = model.startsWith("api_");

  Array.from(attackSelect.options).forEach(option => {
    option.disabled = false;
    if (scenario && !scenario.supported_attacks.includes(option.value)) {
      option.disabled = true;
    }
    if (isApi && (option.value === "fgsm" || option.value === "pgd" || option.value === "contrastive_pgd")) {
      option.disabled = true;
    }
  });
  if (attackSelect.selectedOptions[0]?.disabled) {
    const nextOption = Array.from(attackSelect.options).find(option => !option.disabled);
    if (nextOption) attackSelect.value = nextOption.value;
  }

  document.getElementById("promptTemplateInput").disabled = !isClip;
  document.getElementById("injectionPromptInput").disabled = !(isClip || isApi);
  document.getElementById("goalHijackInput").disabled = !isApi;
  document.getElementById("attackSourceSelect").disabled = !isApi;
  document.getElementById("injectionStrengthSelect").disabled = !isApi;
  document.getElementById("visualModeSelect").disabled = !isApi;
  document.getElementById("visualPositionSelect").disabled = !isApi;
  document.getElementById("visualFontSizeInput").disabled = !isApi;
  document.getElementById("visualOpacityInput").disabled = !isApi;
  document.getElementById("visualContrastInput").disabled = !isApi;
  document.getElementById("queryBudgetInput").disabled = !isApi;
  document.getElementById("universalBudgetInput").disabled = !isApi;
  document.getElementById("systemPromptInput").disabled = !isApi;
  document.getElementById("apiBaseUrlInput").disabled = !isApi;
  document.getElementById("apiModelInput").disabled = !isApi;
  document.getElementById("apiKeyInput").disabled = !isApi;
  document.getElementById("delayedInjectionCheckbox").disabled = !isApi;
  renderScenarioCard();
}

function buildAttackPayload() {
  return {
    dataset_id: document.getElementById("datasetSelect").value,
    scenario: document.getElementById("scenarioSelect").value,
    model_name: document.getElementById("modelSelect").value,
    attack_name: document.getElementById("attackSelect").value,
    epsilon: Number(document.getElementById("epsilonInput").value),
    alpha: Number(document.getElementById("alphaInput").value),
    steps: Number(document.getElementById("stepsInput").value),
    max_samples: Number(document.getElementById("sampleCountInput").value),
    targeted: document.getElementById("targetedCheckbox").checked,
    target_label: document.getElementById("targetTextInput").value.trim() || null,
    prompt_template: document.getElementById("promptTemplateInput").value,
    injection_prompt: document.getElementById("injectionPromptInput").value,
    goal_hijack_instruction: document.getElementById("goalHijackInput").value,
    attack_source: document.getElementById("attackSourceSelect").value,
    injection_strength: document.getElementById("injectionStrengthSelect").value,
    visual_injection_mode: document.getElementById("visualModeSelect").value,
    visual_injection_position: document.getElementById("visualPositionSelect").value,
    visual_font_size: Number(document.getElementById("visualFontSizeInput").value),
    visual_opacity: Number(document.getElementById("visualOpacityInput").value),
    visual_contrast: Number(document.getElementById("visualContrastInput").value),
    delayed_injection: document.getElementById("delayedInjectionCheckbox").checked,
    query_budget: Number(document.getElementById("queryBudgetInput").value),
    universal_budget: Number(document.getElementById("universalBudgetInput").value),
    system_prompt: document.getElementById("systemPromptInput").value,
    api_base_url: document.getElementById("apiBaseUrlInput").value || null,
    api_model: document.getElementById("apiModelInput").value || null,
    api_key: document.getElementById("apiKeyInput").value || null,
  };
}

async function pollRunStatus(runId) {
  const state = await fetchJson(`/api/attacks/runs/${runId}`);
  setProgress(state);
  if (state.status === "completed") {
    clearInterval(currentRunTimer);
    currentRunTimer = null;
    currentRunId = null;
    await loadHistoryAndStats();
    if (state.experiment_id) {
      const detail = await fetchJson(`/api/experiments/${state.experiment_id}`);
      markSelectedExperiment(state.experiment_id);
      renderExperimentDetail(detail);
    }
    document.getElementById("runBtn").disabled = false;
    document.getElementById("runBtn").textContent = "开始实验";
  } else if (state.status === "failed") {
    clearInterval(currentRunTimer);
    currentRunTimer = null;
    currentRunId = null;
    document.getElementById("runBtn").disabled = false;
    document.getElementById("runBtn").textContent = "开始实验";
    alert(state.error || "实验执行失败");
  }
}

async function runAttack() {
  const payload = buildAttackPayload();
  const button = document.getElementById("runBtn");
  button.disabled = true;
  button.textContent = "实验运行中...";
  try {
    const state = await fetchJson("/api/attacks/run-async", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentRunId = state.run_id;
    setProgress(state);
    if (currentRunTimer) {
      clearInterval(currentRunTimer);
    }
    currentRunTimer = setInterval(() => {
      pollRunStatus(currentRunId).catch(error => {
        clearInterval(currentRunTimer);
        currentRunTimer = null;
        currentRunId = null;
        button.disabled = false;
        button.textContent = "开始实验";
        alert(error.message);
      });
    }, 1200);
  } catch (error) {
    button.disabled = false;
    button.textContent = "开始实验";
    alert(error.message);
  }
}

async function clearExperiments() {
  if (!confirm("确认删除当前全部实验记录并清空实验目录？")) {
    return;
  }
  await fetchJson("/api/experiments", { method: "DELETE" });
  document.getElementById("comparisonList").innerHTML = "";
  document.getElementById("metricsCards").innerHTML = "";
  document.getElementById("metricsChart").innerHTML = "";
  document.getElementById("experimentDetail").innerHTML = "<p>尚未选择实验记录。</p>";
  currentSelectedExperimentId = null;
  currentExperimentDetail = null;
  comparisonExpanded = false;
  await loadHistoryAndStats();
}

async function compareExperiments() {
  const ids = document.getElementById("compareIdsInput").value.trim();
  if (!ids) {
    alert("请输入要比较的实验 ID");
    return;
  }
  const payload = await fetchJson(`/api/experiments/compare?ids=${encodeURIComponent(ids)}`);
  renderCompareBoard(payload);
}

async function importDataset() {
  const type = document.getElementById("importTypeSelect").value;
  const datasetId = document.getElementById("importDatasetIdInput").value.trim();
  const datasetName = document.getElementById("importDatasetNameInput").value.trim();
  const imageRoot = document.getElementById("importImageRootInput").value.trim();
  const json1 = document.getElementById("importJsonInput1").value.trim();
  const json2 = document.getElementById("importJsonInput2").value.trim();
  const limit = Number(document.getElementById("importLimitInput").value);
  if (!datasetId || !datasetName || !imageRoot || !json1) {
    alert("请至少填写导入类型、数据集 ID、名称、图片目录和第一个 JSON 路径");
    return;
  }
  const button = document.getElementById("importBtn");
  button.disabled = true;
  button.textContent = "导入中...";
  try {
    if (type === "vqav2" && !json2) {
      throw new Error("VQAv2 导入需要 questions JSON 和 annotations JSON");
    }
    const url = type === "vqav2" ? "/api/datasets/import-vqav2" : "/api/datasets/import-coco";
    const payload = type === "vqav2"
      ? {
          dataset_id: datasetId,
          dataset_name: datasetName,
          image_root: imageRoot,
          questions_json: json1,
          annotations_json: json2,
          limit,
        }
      : {
          dataset_id: datasetId,
          dataset_name: datasetName,
          image_root: imageRoot,
          captions_json: json1,
          limit,
        };
    await fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const overview = await fetchJson("/api/overview");
    document.getElementById("datasetSelect").innerHTML = (overview.datasets || [])
      .map(item => `<option value="${item.dataset_id}">${item.name}</option>`)
      .join("");
    document.getElementById("datasetSelect").value = datasetId;
    await loadSamplesAndSummary();
    alert(`数据集 ${datasetId} 导入完成`);
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "导入数据集";
  }
}

async function init() {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  const [overview, runtimeConfig, catalog] = await Promise.all([
    fetchJson("/api/overview"),
    fetchJson("/api/runtime-config"),
    fetchJson("/api/catalog"),
  ]);
  scenarioCatalog = catalog.scenarios || [];
  renderOverview(overview);
  fillSelect("scenarioSelect", scenarioCatalog, "scenario_id", "name");
  renderModelOptions();
  document.getElementById("attackSelect").innerHTML = (catalog.attacks || [])
    .map(item => `<option value="${item.attack_name}">${item.attack_name}</option>`)
    .join("");
  document.getElementById("datasetSelect").innerHTML = (overview.datasets || [])
    .map(item => `<option value="${item.dataset_id}">${item.name}</option>`)
    .join("");

  const apiDefaults = runtimeConfig.api_defaults || {};
  document.getElementById("apiBaseUrlInput").value = apiDefaults.base_url || "";
  document.getElementById("apiModelInput").value = apiDefaults.model || "";
  if (apiDefaults.api_key_configured) {
    document.getElementById("apiKeyInput").placeholder = `已配置：${apiDefaults.api_key_preview}`;
  }

  syncImportLabels();
  syncFormState();
  resetProgress();
  await loadSamplesAndSummary();
  await loadHistoryAndStats();
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  requestAnimationFrame(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  });
}

document.getElementById("datasetSelect").addEventListener("change", loadSamplesAndSummary);
document.getElementById("scenarioSelect").addEventListener("change", syncFormState);
document.getElementById("modelSelect").addEventListener("change", syncFormState);
document.getElementById("attackSelect").addEventListener("change", syncFormState);
document.getElementById("runBtn").addEventListener("click", runAttack);
document.getElementById("clearExperimentsBtn").addEventListener("click", clearExperiments);
document.getElementById("toggleComparisonBtn").addEventListener("click", () => {
  comparisonExpanded = !comparisonExpanded;
  if (currentExperimentDetail) {
    renderComparisons(currentExperimentDetail);
  }
});
document.addEventListener("change", event => {
  if (!(event.target instanceof HTMLSelectElement)) return;
  if (event.target.id !== "comparisonFilterSelect") return;
  comparisonFilter = event.target.value;
  comparisonExpanded = false;
  if (currentExperimentDetail) {
    renderComparisons(currentExperimentDetail);
  }
});
document.getElementById("compareBtn").addEventListener("click", compareExperiments);
document.getElementById("importBtn").addEventListener("click", importDataset);
document.getElementById("importTypeSelect").addEventListener("change", syncImportLabels);
document.getElementById("experimentHistory").addEventListener("click", event => {
  const target = event.target instanceof Element ? event.target : event.target?.parentElement;
  if (!target) return;
  const card = target.closest(".history-clickable");
  if (!card) return;
  event.preventDefault();
  openExperimentDetail(card.dataset.id);
});
document.getElementById("experimentHistory").addEventListener("keydown", event => {
  const target = event.target instanceof Element ? event.target : event.target?.parentElement;
  if (!target) return;
  const card = target.closest(".history-clickable");
  if (!card) return;
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  openExperimentDetail(card.dataset.id);
});

init();
