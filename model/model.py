import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=1024, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)  # 첫 번째 FC 레이어
        self.fc2 = nn.Linear(512, 256)  # 두 번째 FC 레이어
        self.fc3 = nn.Linear(256, num_classes)  # 출력 레이어
        self.relu = nn.ReLU()  # 활성화 함수

    def forward(self, x):
        x = self.relu(self.fc1(x))  # FC1 → ReLU
        x = self.relu(self.fc2(x))  # FC2 → ReLU
        x = self.fc3(x)  # 최종 출력 (Softmax는 CrossEntropyLoss 내부에서 처리)
        return x

if __name__ == "__main__":
    model = MLP()
    print(model)