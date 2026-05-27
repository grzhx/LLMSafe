# LLMSafe 多模态模型攻击与安全评估平台

## 1. 项目定位

LLMSafe 当前版本面向图像-文本多模态模型的学术实验，重点围绕以下研究目标构建：

- 面向 `CLIP/OpenCLIP` 的零样本分类与图文检索攻击评估
- 面向 `kimi-k2.5` 等兼容 OpenAI Chat Completions 协议模型的视觉问答与图像描述安全评估
- 面向真实 benchmark 的攻击样本生成、实验记录管理、指标统计与结果对比
- 面向提示注入、黑盒查询攻击、梯度攻击与迁移攻击的统一实验流程

当前版本不再暴露未真正落地的音频、视频等占位功能，主线仅保留已经实现的图像-文本场景。

## 2. 已实现能力

### 2.1 实验场景

- `clip_zero_shot_classification`
  - CLIP 零样本图文分类
- `image_text_retrieval`
  - 图文检索与跨模态排序攻击
- `visual_question_answering`
  - 多模态视觉问答提示注入与黑盒攻击
- `image_captioning`
  - 图像描述候选选择与语义偏移分析

### 2.2 攻击方法

- `fgsm`
  - 图像白盒基线
- `pgd`
  - 图像白盒强化基线
- `contrastive_pgd`
  - 面向图文检索的对比损失/排序攻击
- `transfer_pgd`
  - 在本地代理模型上优化后迁移到目标模型
- `prompt_injection`
  - 分层提示注入，支持 `user` / `retrieved_context` / `ocr` / `tool`
- `blackbox_random`
  - 查询预算约束下的黑盒随机扰动攻击

### 2.3 数据集与样本管理

- 内置 `shapes-mm-v1`
  - 仅用于开发调试与回归测试，不作为主实验 benchmark
- 支持导入真实 benchmark
  - `POST /api/datasets/import-coco`
  - `POST /api/datasets/import-vqav2`
- 支持样本摘要、浏览、标注和实验前后图像对比

## 3. 当前主模型

### 3.1 图文对齐模型

- `clip_vit_b32`

用途：

- 零样本分类
- 图文检索
- 对比损失攻击
- 图像扰动与排序退化分析

### 3.2 多模态 API 模型

- `api_openai_compatible`

推荐接入：

- `kimi-k2.5`

需要配置：

- `base_url`
- `api_key`
- `model`

### 3.3 本地代理模型

- `simple_cnn`
- `resnet18_demo`

用途：

- 本地图像 baseline
- 迁移攻击样本生成

## 4. 已实现的真实 benchmark 导入

### 4.1 COCO Captions

接口：

```text
POST /api/datasets/import-coco
```

请求示例：

```json
{
  "dataset_id": "coco-caption-mini-v1",
  "dataset_name": "COCO Caption Mini",
  "image_root": "D:/datasets/coco/val2017",
  "captions_json": "D:/datasets/coco/annotations/captions_val2017.json",
  "limit": 200
}
```

导入后样本包含：

- 正样本 captions
- 检索候选文本 `retrieval_candidates`
- 图像描述候选 `caption_candidates`
- 正样本索引 `positive_indices`

### 4.2 VQAv2

接口：

```text
POST /api/datasets/import-vqav2
```

请求示例：

```json
{
  "dataset_id": "vqav2-mini-v1",
  "dataset_name": "VQAv2 Mini",
  "image_root": "D:/datasets/vqav2/val2014",
  "questions_json": "D:/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
  "annotations_json": "D:/datasets/vqav2/v2_mscoco_val2014_annotations.json",
  "limit": 200
}
```

导入后样本包含：

- 图像
- 问题 `question`
- 主答案 `answer`
- 候选答案 `answer_candidates`
- 答案频次分布 `answer_distribution`

## 5. 评估指标

平台当前输出以下核心指标：

- `attack_success_rate`
- `avg_linf`
- `avg_l2`
- `avg_confidence_shift`
- `avg_semantic_consistency`
- `avg_queries`
- `transfer_success_rate`
- `total_runtime_ms`

检索任务额外输出：

- `retrieval_recall_at_1`
- `retrieval_recall_at_3`
- `mean_rank_shift`
- `mean_average_precision_proxy`

VQA / caption / API 攻击额外输出：

- `answer_shift_rate`
- `constraint_violation_rate`

## 6. Prompt Injection 实验协议

当前提示注入不是“直接把一句错误提示贴在用户问题后面”的简化演示，而是按多模态安全研究的分层协议实现：

1. `system_prompt`
   - 明确模型必须以视觉证据为准
2. `base_prompt`
   - 给定原始任务，例如 VQA 问题或 caption 候选选择任务
3. `attack_source`
   - 指定注入来源：
   - `user`
   - `retrieved_context`
   - `ocr`
   - `tool`
4. `injection_strength`
   - 指定注入强度：`weak` / `medium` / `strong`
5. 结果评估
   - 比较原始回答与攻击后回答
   - 统计 `answer_shift_rate`
   - 统计 `constraint_violation_rate`

这使实验更接近真实多模态系统中的“用户输入 + 外部上下文 + OCR + 工具调用”组合攻击面。

## 7. 主要接口

### 7.1 平台与配置

- `GET /api/overview`
- `GET /api/catalog`
- `GET /api/runtime-config`
- `GET /api/clip/status`
- `GET /api/api/status`

### 7.2 数据集

- `GET /api/datasets`
- `GET /api/datasets/{dataset_id}/samples`
- `GET /api/datasets/{dataset_id}/summary`
- `POST /api/datasets/{dataset_id}/annotate`
- `POST /api/datasets/import-coco`
- `POST /api/datasets/import-vqav2`

### 7.3 实验

- `POST /api/attacks/run`
- `GET /api/experiments`
- `GET /api/experiments/stats`
- `GET /api/experiments/compare`
- `GET /api/experiments/{experiment_id}`

## 8. 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

Windows 推荐启动方式：

```bash
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

访问：

```text
http://127.0.0.1:8000
```

## 9. 模型配置

运行时默认配置文件：

```text
storage/runtime_config.json
```

格式示例：

```json
{
  "api_defaults": {
    "provider": "openai_compatible",
    "base_url": "https://api.moonshot.cn/v1",
    "model": "kimi-k2.5",
    "api_key": "sk-..."
  }
}
```

如果项目刚 fork 下来没有 `storage` 目录，首次启动后会自动创建。也可以手动创建：

- `storage/datasets`
- `storage/models`
- `storage/experiments`
- `storage/runtime_config.json`

## 10. 当前边界

当前版本已经可以支持研究型图像-文本实验，但仍有明确边界：

- 主线仅覆盖图像-文本，不声明音频/视频实验已实现
- `transfer_pgd` 目前要求样本具有可映射到本地代理模型的分类标签
- 真实 benchmark 文件需要用户本地准备后导入
- CLIP 依赖额外安装，并可能需要联网下载权重

这些边界都是已在代码中显式约束的，不再以“占位功能”形式对外展示。
