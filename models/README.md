# 本地模型与当前发布版本

当前服务只有一个七类表情分类器：纯 ViT-S/16 FP32。正式文件由 `server.py` 和 `deployment/runtime-lock.json` 指定；研究目录中的模型不是产品可选项。

| 文件 | 用途 |
|---|---|
| `face_detection_yunet_2023mar.onnx` | OpenCV Zoo YuNet 人脸定位 |
| `vit_emotion_seven_fp32.onnx` | 2026-09-28 早上正式发布的七类表情分类器 |
| `vit_emotion_seven_int8.onnx` | 历史七类基线，当前服务不使用 |
| `research/` 与其他旧模型 | 离线研究或历史对照，不随正式包提供切换 |

正式情绪模型 SHA256：`f9753ef6496707319576794704a0bae7674b46368305edbf366d538b2a7ce54f`。

模型由本机 eNTERFACE 适配第 2/6 轮参数平均得到，运行时只有一次 ViT 推理。骨干来自 timm 的 ImageNet-21K/ImageNet-1K AugReg ViT-S/16，随后以 FERPlus、RAVDESS、CREMA-D、KDEF、ExpW 和 eNTERFACE 数据训练/适配，保留既有的数据及人物/视频划分。输入为扩展方形人脸、224×224 灰度复制三通道、ImageNet 归一化；输出七个 logits，服务做 softmax，摄像头界面另做多帧融合。

类别顺序：anger、disgust、fear、happiness、neutral、sadness、surprise。无蔑视类，无 CNN 分类器选项。离线 CNN 教师候选不代表产品架构变化。

## 当前证据

冻结后打开的最终 ExpW 测试共 2,610 张，准确率 51.80%、宏平均 F1 0.4508；恐惧召回率 15.96%、厌恶 8.20%。eNTERFACE 新人物最终测试共 6 人、180 段，整段准确率 42.78%，连续正确至少半秒的片段占 30.56%。该视频集为受控情绪语音六类，没有中性，不是自然微表情标注。

真实眼镜选择验证共 196 张，准确率 54.08%、宏平均 F1 0.4689，其中恐惧仅 7 张。这不是新的独立测试，也不能保证现场演示效果。旧 INT8 的历史 FERPlus 88.24% 等数字不能用于描述当前模型。

完整逐类精确率、召回率、F1 及部署检查见 [中文测试报告](../output/overnight-validation-20260928/夜间训练与测试结果.md)。原最终测试在 07:43 已打开，后续实验不得重新称其为新盲测。

## 来源

- ViT：[ICLR 2021](https://arxiv.org/abs/2010.11929)。
- YuNet：[论文](https://link.springer.com/article/10.1007/s11633-023-1423-y)、[OpenCV Zoo 模型与 MIT 许可](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- FERPlus：[作者仓库](https://github.com/microsoft/FERPlus)。
- 离线 EmotiEffLib 研究：[作者仓库](https://github.com/sb-ai-lab/EmotiEffLib)。来源、固定版本、许可和 SHA 保留在各研究子目录，它不是当前部署模型。

模型识别可见表情模式，不能测量真实心理状态。光照、姿态、遮挡和个体差异仍会造成误判。
