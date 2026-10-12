"""Create the Chinese handoff outside the deployment archive from measured results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'training/overnight-20260927'
OUT = ROOT/'output/overnight-validation-20260928'
LABELS = dict(zip(('anger','disgust','fear','happiness','neutral','sadness','surprise'),
                  ('愤怒','厌恶','恐惧','开心','中性','悲伤','惊讶')))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct(x):
    return '无样本' if x is None else f'{100*x:.2f}%'


def main():
    final = WORK/'final-heldout'
    decision = read(WORK/'final-selection.json')
    acceptance = read(final/'acceptance.json')
    photos = read(final/'photos-final.json')['results']
    videos = read(final/'videos-final.json')['results']
    matrix = read(WORK/'cpu-candidate-matrix.json')['candidates']['enterface2-6-average-fp32']['variants']
    release = read(WORK/'release-verification.json')
    used = acceptance['release_system']
    assert release['released_system'] == used
    lines = ['# 夜间训练与测试结果（2026-09-28）', '',
        '**本轮没有达到“自然摄像头中七类都很准、细微表情和戴眼镜都能稳定演示”的目标。仍然会错，不能保证现场每种表情都演示成功。**', '',
        ('最终预留测试按预先保存的规则通过，已把所选模型同步到源项目、正在运行的服务和唯一部署包。'
         if acceptance['passed'] else '最终候选未通过预留测试，按预先保存的规则保留原情绪模型。没有用测试集继续调参或更换另一个候选来刷结果。'), '',
        f'正式 ZIP：`{release["zip"]}`', f'ZIP SHA256：`{release["zip_sha256"]}`',
        f'情绪模型 SHA256：`{release["model_sha256"]}`', '',
        '压缩包只包含运行所需文件，不包含这份报告、训练数据、训练环境或演示对照视频。旧正式 ZIP 在新包校验通过后清理。', '',
        '## 预留照片：独立的最后一次测试', '',
        'ExpW 原始来源的 2,610 张预留人脸照片。先固定唯一候选和模型文件 SHA256，再打开测试集。这里的总体准确率是预测正确的照片比例；逐类召回率是该类照片中识别正确的比例。互联网照片不等于你本人、实验室光照下的实时摄像头准确率。', '',
        '| 系统 | 总体准确率 | 七类宏平均 F1 |', '|---|---:|---:|']
    for name, title in [('baseline','旧模型'),('selected','最终候选')]:
        r = photos[name]; lines.append(f'| {title} | {pct(r["accuracy"])} | {r["macro_f1"]:.4f} |')
    lines += ['', '| 表情 | 样本数 | 旧召回率 | 候选召回率 | 候选精确率 | 候选 F1 |', '|---|---:|---:|---:|---:|---:|']
    for c, title in LABELS.items():
        a, b = photos['baseline']['per_class'][c], photos['selected']['per_class'][c]
        lines.append(f'| {title} | {b["support"]} | {pct(a["recall"])} | {pct(b["recall"])} | {pct(b["precision"])} | {b["f1"]:.4f} |')
    lines += ['', '**恐惧在这次预留照片测试中退步，厌恶仍然很弱。总体分数提升不能解释为每一类都提升。**', '',
        '## 预留视频：新人物测试', '',
        'eNTERFACE 的 6 名预留人物、180 段视频，每类 30 段，人物未进入训练或选择验证。该数据是受控情绪语音，有六类，没有中性。片段情绪标签不等于每帧的面部表情真值，也不是自然微表情数据。', '',
        '| 系统 | 整段平均预测正确 | 六类宏平均 F1 | 连续显示正确至少 0.5 秒的片段 |', '|---|---:|---:|---:|']
    for name, title in [('baseline','旧模型'),('selected','最终候选')]:
        r = videos[name]; lines.append(f'| {title} | {pct(r["accuracy"])} | {r["macro_f1_present"]:.4f} | {pct(r["stable_correct_half_second_clip_rate"])} |')
    lines += ['', '| 表情 | 候选整段召回率 | 候选精确率 | 候选 F1 | 候选连续 0.5 秒成功率 |', '|---|---:|---:|---:|---:|']
    for c, title in LABELS.items():
        b = videos['selected']['per_class'][c]
        if b['support']:
            lines.append(f'| {title} | {pct(b["recall"])} | {pct(b["precision"])} | {b["f1"]:.4f} | {pct(b["stable_correct_half_second_clip_rate"])} |')
    lines += ['', '## 戴眼镜与普通强度表情：用于选模型的验证', '',
        '以下验证集已用于选择候选，不是新的最终盲测。真实眼镜照片 196 张；恐惧只有 7 张，无法给出可靠的人群准确率。眼镜视频来自两个验证人物，每种表情共 10 段，人物和片段之间也有关联。', '',
        '| 项目 | 旧模型 | 候选 |', '|---|---:|---:|']
    for title, key in [('真实眼镜照片准确率','accuracy'),('真实眼镜照片宏平均 F1','macro_f1')]:
        a, b = (matrix[t]['photos']['real_eyeglasses_expanded'][key] for t in ('baseline','raw'))
        lines.append(f'| {title} | {pct(a) if key=="accuracy" else f"{a:.4f}"} | {pct(b) if key=="accuracy" else f"{b:.4f}"} |')
    for c in ('anger','fear'):
        a,b=(matrix[t]['rav_normal']['per_class'][c]['stable_correct_half_second_clip_rate'] for t in ('baseline','raw'))
        lines.append(f'| 普通强度{LABELS[c]}连续半秒（各 16 段） | {round(a*16)}/16 | {round(b*16)}/16 |')
    for c in ('anger','fear'):
        a,b=(matrix[t]['enterface_groups']['eyewear']['per_class'][c]['stable_correct_half_second_clip_rate'] for t in ('baseline','raw'))
        lines.append(f'| 戴眼镜{LABELS[c]}连续半秒（各 10 段） | {round(a*10)}/10 | {round(b*10)}/10 |')
    lines += ['', '自然场景 CAER 的固定清晰帧外观诊断：152 帧标签一致率由 59.87% 降至 53.29%；有样本类别宏平均 F1 由 0.3131 升至 0.3224。这是助手辅助的探索性外观标注，没有可靠恐惧样本，不能作为自然场景七类准确率。退步已保留，没有重新标注来提高分数。', '',
        '## 实际训练与取舍', '',
        '整夜完成多次视觉模型训练，包括原始 ExpW 扩充、眼镜加权、RGB、情绪语音视频、DINOv2 16 轮，以及本机三张反馈照片的 4 轮分类头训练。另有两次 30 轮的小型概率序列模型训练；这不是完整视觉 Transformer 又训练了 60 轮。', '',
        '最终候选是一份 ViT-S/16 FP32 模型：eNTERFACE 适配训练的第 2 和第 6 轮参数各占一半。参数平均不算一次新增训练，运行时也只推理一个模型。', '',
        'DINOv2 的大样本照片表现更好，但严格眼镜小样本未达标；独立人物视频的恐惧很弱，与其他候选的固定混合也失去眼镜恐惧的持续识别成功片段。因此保留研究结果，没有把它直接放进产品。两次时序尝试也因恐惧退步被拒绝。三张反馈照片属于训练数据，其命中不能算独立准确率。', '',
        '本轮使用可公开研究使用的 ExpW、eNTERFACE 等来源，并保留下载和标签来源、人物/视频划分。曾发现一个 RAF 镜像标签映射错误，其分数已作废，且授权不明的数据没有用于这次发布。FERPlus 训练/验证精确重复已排查并从选择验证中剔除。原始测试数据没有加入训练。', '',
        '## 运行与交付检查', '',
        '七分类、紧人脸框、隐藏阈值、最近人脸的语音开关保留。环境安装在解压项目自身的 `.runtime`，不要求安装训练环境。新电脑若缺微软 C++ 运行库，安装器可能需要系统确认；本机测试不能替代另一台电脑的实际硬件检查。', '',
        f'交付检查记录：`{WORK / "release-verification.json"}`',
        '浏览器摄像头测试使用固定视频作为虚拟摄像头，调用真实本地 API，语音使用静音记录器检查播报对象与开关。这验证功能与延迟，不代表验证了物理摄像头、扬声器或新人物准确率。', '',
        f'旧版与候选的固定片段对照：`{OUT / "old-vs-selected-validation.mp4"}`。19 个片段按事先固定顺序选取，包含失败与判断中，不是只挑成功案例。', '',
        f'最终选型记录：`{WORK / "final-selection.json"}`', f'最终测试与规则判定：`{final}`', '',
        '自动审批曾拦截删除临时解压测试副本，理由只有 “blocked by policy”。副本暂留 output，未换方式绕过删除；正式 dist 只保留一个 ZIP。', '']
    OUT.mkdir(exist_ok=True)
    path = OUT/'夜间训练与测试结果.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
