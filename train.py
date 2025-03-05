import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import CNN
from utils import ImageDataset
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Classification Training Script')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률 (기본값: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='가중치 감쇠 (기본값: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기 (기본값: 32)')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에포크 수 (기본값: 5)')

    # 최적화 관련 파라미터
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'], help='옵티마이저 선택 (기본값: Adam)')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='손실 함수 (기본값: CrossEntropyLoss)')
    
    # 하드웨어 관련 파라미터
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='학습에 사용할 장치 (기본값: 자동 감지)')

    # Datafiles path
    parser.add_argument('--train_csv', type=str, default='./datafiles/train.csv', help='훈련 데이터 CSV 파일 경로 (기본값: ./train.csv)')
    parser.add_argument('--test_csv', type=str, default='./datafiles/test.csv', help='테스트 데이터 CSV 파일 경로 (기본값: ./test.csv)')
    parser.add_argument('--submission_csv', type=str, default='./datafiles/submission.csv', help='제출용 CSV 파일 경로 (기본값: ./submission.csv)')
    
    return parser.parse_args()

def train(model, dataloader, criterion, optimizer, epochs=50, device='cuda'):
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

if __name__ == "__main__":
    # 하이퍼파라미터 파싱
    args = parse_arguments()
    
    label2id = {'airplane': 0, 'apple': 1, 'ball': 2, 'bird': 3, 'building': 4, 'cat': 5, 'emotion_face': 6, 'police_car': 7, 'rabbit': 8, 'truck': 9}
    id2label = {v: k for k, v in label2id.items()}
    print(id2label)

    # Dataset & DataLoader
    train_csv_path = args.train_csv
    test_csv_path = args.test_csv
    
    train_dataset = ImageDataset(train_csv_path, mode='train')
    test_dataset = ImageDataset(test_csv_path, mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 장치 설정
    device = torch.device(args.device)
    
    # Model, Criterion, Optimizer
    model = CNN()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저 선택
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 학습 및 평가
    train(model, train_dataloader, criterion, optimizer, epochs=args.epochs, device=device)
    
    # id2label
    id2label = {
        0: 'airplane', 1: 'apple', 2: 'ball', 3: 'bird', 4: 'building',
        5: 'cat', 6: 'emotion_face', 7: 'police_car', 8: 'rabbit', 9: 'truck'
    }
    
    # 모델을 평가 모드로 전환
    model.eval()
    
    # 예측값을 저장할 리스트
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
    submission_df = pd.read_csv(args.submission_csv)
    
    # 예측된 결과를 'label' 열에 추가
    submission_df['label'] = predictions
    
    # 결과를 기존 submission.csv 파일에 덮어쓰기
    submission_df.to_csv(args.submission_csv, index=False)
    
    print("Submission updated and saved as 'submission.csv'.")