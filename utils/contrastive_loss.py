import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import LabelSmoothCrossEntropyLoss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    @staticmethod
    def forward(audio_output, visual_output, text_output):
        audio_visual_sim = F.cosine_similarity(audio_output.unsqueeze(2), visual_output.unsqueeze(1), dim=-1)
        audio_text_sim = F.cosine_similarity(audio_output.unsqueeze(2), text_output.unsqueeze(1), dim=-1)
        visual_text_sim = F.cosine_similarity(visual_output.unsqueeze(2), text_output.unsqueeze(1), dim=-1)

        # audio_visual_loss = -audio_visual_sim.mean()
        # audio_text_loss = -audio_text_sim.mean()
        # visual_text_loss = -visual_text_sim.mean()
        audio_visual_loss = torch.clamp(audio_visual_sim.mean(), min=0)
        audio_text_loss = torch.clamp(audio_text_sim.mean(), min=0)
        visual_text_loss = torch.clamp(visual_text_sim.mean(), min=0)

        total_loss = audio_visual_loss + audio_text_loss + visual_text_loss

        return total_loss


class CombinedLoss(nn.Module):
    def __init__(self, smoothing=0.0, temperature=0.1, weight=None, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.label_smooth_loss = LabelSmoothCrossEntropyLoss(weight=weight, reduction=reduction, smoothing=smoothing)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, inputs, targets, audio_output, visual_output, text_output):
        label_loss = self.label_smooth_loss(inputs, targets)
        contrastive_loss = self.contrastive_loss(audio_output, visual_output, text_output)
        total_loss = label_loss + contrastive_loss
        return total_loss


# 使用示例
if __name__ == '__main__':
    # 创建损失函数实例
    combined_loss = CombinedLoss(smoothing=0.1, temperature=0.1)
    # 假设有一些输入和目标
    inputs = torch.randn(24, 10)  # 示例输入
    targets = torch.randint(0, 10, (24,))  # 示例目标
    audio_output = torch.randn(24, 512, 768)  # 示例音频输出
    visual_output = torch.randn(24, 196, 768)  # 示例视觉输出
    text_output = torch.randn(24, 80, 768)  # 示例文本输出

    # 计算组合损失
    loss = combined_loss(inputs, targets, audio_output, visual_output, text_output)
    print(loss)
