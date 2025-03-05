import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import MLP
from utils import ImageDataset
import pandas as pd
import argparse
import numpy as np
import copy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ensemble Image Classification Training Script')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률 (기본값: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='가중치 감쇠 (기본값: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기 (기본값: 32)')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에포크 수 (기본값: 5)')
    parser.add_argument('--num_models', type=int, default=5, help='앙상블할 모델 개수 (기본값: 5)')

    # 최적화 관련 파라미터
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'], help='옵티마이저 선택 (기본값: Adam)')
    
    # 하드웨어 관련 파라미터
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='학습에 사용할 장치 (기본값: 자동 감지)')
    
    return parser.parse_args()

def train_single_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs, device):
    model.train()
    best_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0 
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = 100 * correct / total
        
        # 최고 성능 모델 상태 저장
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        print(f"Model Training - Epoch [{epoch+1}/{epochs}] Accuracy: {epoch_acc:.2f}% Loss: {epoch_loss:.2f}%")
    
    return best_model_state, best_accuracy

def ensemble_predict(models, test_dataloader, device, id2label):
    predictions = []
    
    # 모든 모델의 예측 결과 수집
    all_outputs = []
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            
            # 각 모델의 출력 수집
            batch_outputs = []
            for model in models:
                model.eval()
                output = model(inputs)
                batch_outputs.append(output.cpu().numpy())
            
            # 배치별 모델 출력 평균
            batch_ensemble_output = np.mean(batch_outputs, axis=0)
            predicted = np.argmax(batch_ensemble_output, axis=1)
            
            # 클래스 이름으로 변환
            for idx in predicted:
                label = id2label[idx]
                predictions.append(label)
    
    return predictions

def main():
    # 하이퍼파라미터 파싱
    args = parse_arguments()
    
    # 레이블 매핑
    label2id = {'airplane': 0, 'apple': 1, 'ball': 2, 'bird': 3, 'building': 4, 'cat': 5, 'emotion_face': 6, 'police_car': 7, 'rabbit': 8, 'truck': 9}
    id2label = {v: k for k, v in label2id.items()}
    
    # 장치 설정
    device = torch.device(args.device)
    
    # Dataset & DataLoader
    train_csv_path = "./train.csv"
    test_csv_path = "./test.csv"
    
    train_dataset = ImageDataset(train_csv_path, mode='train')
    test_dataset = ImageDataset(test_csv_path, mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 학습된 모델들 저장할 리스트
    ensemble_models = []
    
    # 여러 모델 학습
    for i in range(args.num_models):
        print(f"\n--- Training Model {i+1}/{args.num_models} ---")
        
        # 모델 및 최적화 도구 초기화
        model = MLP().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저 선택
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 개별 모델 학습
        best_model_state, best_accuracy = train_single_model(
            model, 
            train_dataloader, 
            test_dataloader, 
            criterion, 
            optimizer, 
            epochs=args.epochs, 
            device=device
        )
        
        # 최고 성능 모델 상태 로드
        model.load_state_dict(best_model_state)
        ensemble_models.append(model)
        
        print(f"Model {i+1} Best Accuracy: {best_accuracy:.2f}%")
    
    # 앙상블 예측
    predictions = ensemble_predict(ensemble_models, test_dataloader, device, id2label)
    
    # 제출 파일 업데이트
    submission_df = pd.read_csv('submission.csv')
    submission_df['label'] = predictions
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nEnsemble Submission updated and saved as 'submission.csv'.")

if __name__ == "__main__":
    main()