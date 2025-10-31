# Notebook 使用指南

本项目提供两个版本的 Jupyter Notebook，分别适用于不同的平台。

## 📓 Colab 版本 vs Kaggle 版本

### Google Colab 版本 (`HybridMedNet_Colab.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alltobebetter/HybridMedNet/blob/main/HybridMedNet_Colab.ipynb)

**优点：**
- ✅ 快速启动，无需注册
- ✅ 免费 GPU (T4)
- ✅ 与 Google Drive 集成
- ✅ 易于分享和协作

**缺点：**
- ⚠️ 不能后台运行（关闭浏览器会断开）
- ⚠️ 会话时长限制（12小时）
- ⚠️ 需要手动上传 Kaggle API key
- ⚠️ 空闲超时（90分钟）

**适合场景：**
- 快速测试和实验
- 短时间训练（< 2小时）
- 需要频繁交互

---

### Kaggle 版本 (`HybridMedNet_Kaggle.ipynb`)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/alltobebetter/HybridMedNet/blob/main/HybridMedNet_Kaggle.ipynb)

**优点：**
- ✅ **可以后台运行**（关闭浏览器继续训练）
- ✅ 免费 GPU (P100/T4)，性能更好
- ✅ 直接访问 Kaggle 数据集，无需下载
- ✅ 自动保存输出文件
- ✅ 版本管理（每次运行保存一个版本）
- ✅ 更长的运行时间（最长 12 小时）

**缺点：**
- ⚠️ 需要 Kaggle 账号
- ⚠️ 需要手动添加数据集

**适合场景：**
- **长时间训练（推荐）**
- 需要后台运行
- 使用 Kaggle 数据集
- 需要版本管理

---

## 🚀 快速开始

### Colab 版本

1. 点击上方的 "Open In Colab" 按钮
2. 运行第一个 cell 检查 GPU
3. 设置 `USE_REAL_DATASET = True`（如果使用真实数据）
4. 上传 `kaggle.json`（如果使用真实数据）
5. 点击 `Runtime` → `Run all`

### Kaggle 版本（推荐用于长时间训练）

1. 点击上方的 Kaggle 按钮
2. 在 Kaggle 中打开 notebook
3. 点击右侧 **"Add data"**
4. 搜索 **"chest-xray-pneumonia"**
5. 添加 **"paultimothymooney/chest-xray-pneumonia"**
6. 设置 GPU: `Settings` → `Accelerator` → `GPU`
7. 点击 **"Save Version"** → **"Save & Run All"**
8. **可以关闭浏览器，训练会继续进行**

---

## 📊 性能对比

| 特性 | Colab | Kaggle |
|------|-------|--------|
| GPU | T4 | P100/T4 |
| 后台运行 | ❌ | ✅ |
| 最长运行时间 | 12小时 | 12小时 |
| 空闲超时 | 90分钟 | 无 |
| 数据集访问 | 需下载 | 直接挂载 |
| 输出保存 | 手动 | 自动 |
| 版本管理 | ❌ | ✅ |

---

## 🎯 推荐使用场景

### 使用 Colab 如果你：
- 想快速测试代码
- 训练时间 < 2 小时
- 需要频繁调试
- 习惯使用 Google Drive

### 使用 Kaggle 如果你：
- **需要长时间训练（30 epochs ≈ 60-90 分钟）**
- 想要后台运行
- 使用 Kaggle 数据集
- 需要保存多个训练版本

---

## 💡 最佳实践

### Colab
```python
# 快速测试配置
config.TRAIN['epochs'] = 10
config.TRAIN['batch_size'] = 16
```

### Kaggle
```python
# 完整训练配置
config.TRAIN['epochs'] = 30
config.TRAIN['batch_size'] = 32

# 点击 "Save & Run All" 后可以关闭浏览器
# 训练完成后会收到邮件通知
```

---

## 📝 数据集说明

### Colab
需要上传 `kaggle.json` 来下载数据集：
1. 访问 https://www.kaggle.com/settings
2. 点击 "Create New API Token"
3. 下载 `kaggle.json`
4. 在 Colab 中上传

### Kaggle
直接添加数据集，无需下载：
1. 点击右侧 "Add data"
2. 搜索并添加数据集
3. 自动挂载到 `/kaggle/input/`

---

## 🔧 故障排除

### Colab
- **GPU 不可用**: `Runtime` → `Change runtime type` → `GPU`
- **会话断开**: 保持浏览器标签页打开
- **内存不足**: 降低 `batch_size`

### Kaggle
- **数据集未找到**: 检查是否添加了数据集
- **GPU 不可用**: `Settings` → `Accelerator` → `GPU`
- **输出文件丢失**: 检查 `/kaggle/working/` 目录

---

## 📧 获取训练结果

### Colab
- 手动下载 `best_model.pth`
- 或保存到 Google Drive

### Kaggle
- 自动保存在 Output 标签页
- 可以直接下载或分享链接
- 每个版本都会保存

---

## 🎓 总结

**快速测试 → 使用 Colab**  
**正式训练 → 使用 Kaggle** ⭐

两个版本的代码完全相同，只是针对不同平台做了优化。选择最适合你需求的版本即可！
