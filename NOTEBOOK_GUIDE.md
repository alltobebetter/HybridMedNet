# Notebook 使用指南

本项目提供 Google Colab Notebook，可以免费使用 GPU 进行训练。

## 📓 Google Colab 版本

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



## 🚀 快速开始

### Colab 版本

1. 点击上方的 "Open In Colab" 按钮
2. 运行第一个 cell 检查 GPU
3. 设置 `USE_REAL_DATASET = True`（如果使用真实数据）
4. 上传 `kaggle.json`（如果使用真实数据）
5. 点击 `Runtime` → `Run all`

---

## 💡 使用建议

### 训练配置
```python
# 快速测试（10 epochs，约 20 分钟）
config.TRAIN['epochs'] = 10
config.TRAIN['batch_size'] = 16

# 完整训练（30 epochs，约 60-90 分钟）
config.TRAIN['epochs'] = 30
config.TRAIN['batch_size'] = 32
```

### 注意事项
- 保持浏览器标签页打开（Colab 会话会在关闭后断开）
- 建议使用 Google Drive 保存模型
- 训练时间较长时，可以使用 Colab Pro（后台运行）

---

## 📝 数据集说明

需要上传 `kaggle.json` 来下载数据集：
1. 访问 https://www.kaggle.com/settings
2. 点击 "Create New API Token"
3. 下载 `kaggle.json`
4. 在 Colab 中上传

---

## 🔧 故障排除

- **GPU 不可用**: `Runtime` → `Change runtime type` → `GPU`
- **会话断开**: 保持浏览器标签页打开
- **内存不足**: 降低 `batch_size`
- **依赖冲突**: 重启 runtime 并重新运行

---

## 📧 获取训练结果

- 手动下载 `best_model.pth`
- 或保存到 Google Drive
- 查看训练曲线和评估结果

---

## 🎓 总结

使用 Google Colab 可以免费训练医学影像分类模型，预期准确率 90%+。完整训练大约需要 60-90 分钟。
