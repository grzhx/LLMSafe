# LLMSafe 多模态模型攻击与安全评估平台

## 项目简介

LLMSafe 是一个面向多模态模型攻击与安全评估的实验平台，当前聚焦于“图像 + 文本”场景，围绕图文对齐模型与多模态大模型 API 的安全性验证，提供攻击配置、攻击样本生成、实验记录管理、指标统计分析和可视化展示能力。

本项目优先实现以下两条主线：

- 基于 CLIP 零样本图文分类的白盒/黑盒图像攻击
- 基于多模态 API 的提示注入攻击与黑盒扰动攻击

平台适用于课程设计、科研原型验证和多轮攻击实验复现。

---

## 项目目标

本项目围绕“多模态模型攻击与安全评估平台”的建设目标，重点实现以下功能：

1. 攻击场景设计  
支持图像分类、图文零样本分类、多模态 API 图像理解等场景下的攻击实验。

2. 多模态数据集与样本管理  
统一管理图像样本、标签、描述文本、问题与答案，并保存攻击前后图像及实验记录。

3. 攻击算法实现  
支持基于梯度优化的图像对抗扰动生成、提示注入攻击、黑盒随机扰动攻击。

4. 攻击效果评估与鲁棒性分析  
输出攻击成功率、扰动强度、置信度偏移、语义一致性、查询次数、迁移成功率等指标。

5. 平台展示  
提供前端页面用于攻击参数配置、样本浏览、实验运行、结果查询与可视化分析。

---

## 当前实现内容

### 1. 多模态数据集

项目内置一个演示数据集：

- 数据集名称：`Synthetic Shapes Multimodal Demo`
- 数据集 ID：`shapes-mm-v1`
- 模态：图像 + 文本
- 样本类别：
  - `circle`
  - `square`
  - `triangle`
  - `cross`

每条样本包含：

- 图像路径
- 标签
- 图像描述文本 `caption`
- 问题 `question`
- 答案 `answer`

数据集用于模拟：

- CLIP 零样本图文分类
- 多模态 API 图像理解
- 提示注入与黑盒攻击实验

---

## 2. 支持的目标模型

### 本地图文对齐/代理模型

- `clip_vit_b32`
  - 用于 CLIP 零样本图文分类攻击
  - 需要额外安装 `open_clip_torch` 或 `transformers`

- `simple_cnn`
  - 本地代理分类模型
  - 用于基线攻击和迁移攻击

- `resnet18_demo`
  - 本地代理分类模型
  - 用于黑盒代理与迁移实验

### 多模态 API 目标

- `api_mock_vision`
  - 本地模拟多模态 API
  - 不依赖外部网络
  - 用于离线演示提示注入和黑盒攻击流程

- `api_openai_compatible`
  - 面向兼容 OpenAI Chat Completions 协议的多模态 API
  - 当前已配置为 Moonshot 官方接口
  - 默认模型：`kimi-k2.5`

---

## 3. 支持的攻击方式

### FGSM

单步梯度符号攻击，用于快速生成图像对抗样本。

适用对象：

- `clip_vit_b32`
- `simple_cnn`
- `resnet18_demo`

### PGD

多步投影梯度攻击，相比 FGSM 更强。

适用对象：

- `clip_vit_b32`
- `simple_cnn`
- `resnet18_demo`

### Prompt Injection

通过在文本提示中加入诱导性指令，使模型忽略图像真实语义，输出指定类别。

适用对象：

- `clip_vit_b32`
- `api_mock_vision`
- `api_openai_compatible`

### BlackBox Random

通过多次随机扰动和查询，寻找可使模型输出偏移的图像输入，适合黑盒攻击实验。

适用对象：

- `clip_vit_b32`
- `api_mock_vision`
- `api_openai_compatible`
- `simple_cnn`
- `resnet18_demo`

---

## 4. 平台评估指标

平台当前支持以下指标统计：

- `attack_success_rate`：攻击成功率
- `linf`：最大扰动强度
- `l2`：整体扰动强度
- `mse`：均方误差
- `confidence_shift`：模型输出置信度偏移
- `semantic_consistency`：语义一致性估计
- `queries`：黑盒查询次数
- `transfer_success_rate`：迁移攻击成功率
- `elapsed_ms`：样本级实验耗时

---

## 系统架构

项目主要由三部分组成：

### 1. 后端

基于 FastAPI 实现，负责：

- 数据集加载与样本管理
- 模型调度
- 攻击算法执行
- API 目标调用
- 实验结果存储
- 前端接口提供

核心文件：

- `app/main.py`
- `app/clip_adapter.py`
- `app/api_adapter.py`

### 2. 前端

基于原生 HTML / CSS / JavaScript 实现，支持：

- 目标模型选择
- 攻击算法选择
- 参数配置
- 样本浏览
- 攻击前后对比
- 指标可视化
- 实验历史查看

核心文件：

- `app/static/index.html`
- `app/static/app.js`
- `app/static/styles.css`

### 3. 存储层

用于保存：

- 数据集元信息
- 样本文件
- 模型权重
- 实验结果
- 默认 API 配置

目录：

- `storage/datasets/`
- `storage/models/`
- `storage/experiments/`
- `storage/runtime_config.json`

---

## 运行环境

建议环境：

- Python 3.11
- Windows / Linux / macOS 均可

当前依赖：

- `fastapi`
- `uvicorn`
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `Pillow`
- `requests`

安装依赖：

```bash
pip install -r requirements.txt
```

如果要启用 CLIP：

```bash
pip install open_clip_torch
```

或者：

```bash
pip install transformers
```

---

## 启动方式

在项目根目录执行：

```bash
python -m uvicorn app.main:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000
```

---

## 默认 API 配置

当前默认多模态 API 配置保存在：

- `storage/runtime_config.json`

默认已配置：

- `base_url = https://api.moonshot.cn/v1`
- `model = kimi-k2.5`

说明：

- 页面会自动读取默认 `base_url` 和 `model`
- `api_key` 只保存在本地配置中，前端仅显示脱敏预览
- 仍可在页面手动覆盖这些配置

---

## 实验流程

一次完整实验的执行流程如下：

1. 选择数据集
2. 选择目标模型
3. 选择攻击方式
4. 设置攻击参数
5. 提交实验请求
6. 后端生成攻击样本或构造攻击提示
7. 调用模型或 API 获得攻击结果
8. 统计攻击指标
9. 保存实验记录
10. 前端展示图像对比与指标结果

---

## 目录结构

```text
LLMSafe/
├─ app/
│  ├─ main.py
│  ├─ clip_adapter.py
│  ├─ api_adapter.py
│  └─ static/
│     ├─ index.html
│     ├─ app.js
│     └─ styles.css
├─ storage/
│  ├─ datasets/
│  ├─ models/
│  ├─ experiments/
│  └─ runtime_config.json
├─ requirements.txt
└─ README.md
```

---

## 当前已验证能力

已完成验证：

- 本地代理模型攻击流程可运行
- `api_mock_vision` 的提示注入与黑盒攻击可运行
- Moonshot 官方 API `kimi-k2.5` 可进行真实图像请求测试
- 平台已修复 Moonshot `temperature` 参数导致的 400 错误

当前注意事项：

- `clip_vit_b32` 需要本地额外安装 CLIP 相关依赖后才能运行
- 使用真实 API 时，需要保证账户额度、模型权限和网络访问正常

---

## 后续可扩展方向

后续可继续扩展：

- 图文检索攻击
- 视觉问答攻击
- 图像描述攻击
- 多模型对比实验
- 更多黑盒迁移攻击算法
- 批量实验与参数扫描
- 更丰富的图表与导出功能

---

## 项目说明

本项目当前是一个“多模态攻击平台原型系统”，强调：

- 攻击流程标准化
- 样本管理规范化
- 实验结果可复现
- 指标分析可视化

适合作为多模态模型攻击与安全评估方向的课程设计、毕业设计或科研原型基础。
