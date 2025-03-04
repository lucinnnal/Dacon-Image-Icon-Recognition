import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import MLP
from utils import ImageDataset
import pandas as pd

def train(model, dataloader, criterion, optimizer, epochs=50):
    model.train()  # 학습 모드
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  

            # 순전파
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 계산
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

def evaluate(model, dataloader):
    model.eval()  # 평가 모드
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.armax(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    label2id = {'airplane': 0, 'apple': 1, 'ball': 2, 'bird': 3, 'building': 4, 'cat': 5, 'emotion_face': 6, 'police_car': 7, 'rabbit': 8, 'truck': 9}
    id2label = {v: k for k, v in label2id.items()}
    print(id2label)
    # Dataset & DataLoader
    train_csv_path = "./train.csv"
    test_csv_path = "./test.csv"

    train_dataset = ImageDataset(train_csv_path, mode = 'train')
    test_dataset = ImageDataset(test_csv_path, mode = 'test')

    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = MLP()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train(model, train_dataloader, criterion, optimizer, epochs=8)

    # id2label
    id2label = {
        0: 'airplane', 1: 'apple', 2: 'ball', 3: 'bird', 4: 'building',
        5: 'cat', 6: 'emotion_face', 7: 'police_car', 8: 'rabbit', 9: 'truck'
    }

    # 모델을 평가 모드로 전환
    model.eval()

    #    예측값을 저장할 리스트
    predictions = []

    # 테스트 데이터셋에 대한 예측 수행
    with torch.no_grad():  # 기울기 계산 안 함
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)  # GPU로 이동 (device 설정이 필요)
        
            # 모델에 입력
            outputs = model(inputs)
        
            # 예측된 클래스 ID
            predicted = torch.argmax(outputs, 1)
        
            # 예측된 클래스 ID를 class 이름으로 변환
            for idx in range(predicted.size(0)):
                label = id2label[predicted[idx].item()]
                predictions.append(label)

    # 기존 submission.csv 파일 읽기
    submission_df = pd.read_csv('submission.csv')

    # 예측된 결과를 'label' 열에 추가
    submission_df['label'] = predictions

    # 결과를 기존 submission.csv 파일에 덮어쓰기
    submission_df.to_csv('submission.csv', index=False)

    print("Submission updated and saved as 'submission.csv'.")