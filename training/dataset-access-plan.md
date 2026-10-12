# 可用视频数据与获取条件

2026-09-18 后续实际执行：已重新寻找并取得 MELD 作者公开原视频，SHA256 校验后抽帧核查，发现对白情绪标签经常不能对应当帧人脸，**未用于训练**。另取得七类 RAF-DB/AffectNet 混合镜像，核对五个 Parquet 的发布端 SHA256，按原 RAF-DB 测试划分做一次 ViT 微调；实验详见 `runs/vit-small-seven-rafdb-v1/experiment-report.md`。其新权重在 RAF 愤怒明显改善，但 RAVDESS、CREMA-D 与眼镜场景退步，故**未部署**。这些普通表情图片不是带时间标注的真正微表情数据；镜像不声明原始数据许可，源图只留本机研究。

2026-09-18 已查原始发布方，不把第三方镜像的许可证声明当作原始数据授权。

1. [FERV39k 官方仓库](https://github.com/wangyanckxx/FERV39k)：七类自然场景表情视频，适合先提升自然视频泛化，但不是严格的微表情数据。官方条款限非商业研究，要求在下载前从有效大学邮箱给原作者发申请邮件，写明姓名、单位、部门、职务和用途；原始视频不能公开分发。Hugging Face 上虽有压缩包镜像，也不能跳过原作者的申请条件。
2. [MMEW 官方仓库](https://github.com/benxianyeteam/MMEW-Dataset)：约 300 段真正的微表情序列和 900 段宏表情，涵盖开心、惊讶、愤怒、厌恶、恐惧、悲伤六种，不含现行系统的独立“中性”微表情类别。官方要求签署 [数据协议](https://github.com/benxianyeteam/MMEW-Dataset/blob/c5f0d04a8a64a3d7d30d40b9e299c591b3268231/MMEW%20release%20agreement.pdf)，把扫描件发给作者后获取数据。需要按人物隔离训练/验证/测试，并把中性基线作为视频状态单独建模。
3. 第三方 [Micro-Expression-Recognition 压缩包](https://huggingface.co/datasets/GriErik123/Micro-Expression-Recognition)：只用 HTTP 范围请求读取约 3.4 MB 的归档目录，未下载 4.1 GB 原始图片。389740 条归档项主要是 CASME II/SAMM 的增强图，显式训练标签是 negative、positive、surprise 三种；不能直接当七类数据，原始 CASME II/SAMM 使用条件也不能由镜像声明替代。

已用现有 RAVDESS/CREMA-D 做两帧实验，见 `runs/temporal-v1/experiment-report.md`。结果没有达到愤怒、恐惧和低强度三项同时改善，因此系统保持现行模型。要推进正式训练，需获得原作者许可的数据，并收集实验室相机里带中性起点、表情过程及正确标签的视频。训练数据只放本机，不放进给实验室的 ZIP。

FERV39k 官方申请邮件需要由学校邮箱发出，不能由项目自动代发。可用以下内容填写真实信息后发送给仓库所列作者：

> Subject: (fudanroilab) Application to download the FERV39k Dataset
>
> Name: [real name]
>
> Affiliation: [university]
>
> Department: [department]
>
> Position: [student/researcher]
>
> Email: [university email]
>
> I have read and agree to the terms and conditions specified on the FERV39k database webpage. The dataset will be used solely for non-commercial academic research on facial expression recognition in our laboratory. I will not share or distribute the dataset or use it commercially.

MMEW 必须先由使用者查看并签署官方协议，再发给仓库指定作者。不能由模型生成的文本代替签署。
