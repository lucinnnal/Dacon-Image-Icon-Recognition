import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, csv_path, target_column=1, data_start_column=2, mode='train', image_size=32):
        self.df = pd.read_csv(csv_path)
        self.target_column = target_column
        self.data_start_column = data_start_column
        self.mode = mode
        self.image_size = image_size

        # Label Map 생성 (유니크한 라벨을 인덱스로 변환)
        unique_labels = sorted(self.df.iloc[:, self.target_column].unique())  # 정렬하여 일정한 인덱스 유지
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # Get label and apply label map
            label = self.df.iloc[idx, self.target_column]
            label_idx = self.label_map[label]  # 매핑 적용

            # Get image data
            image_data = self.df.iloc[idx, self.data_start_column:].values.astype(np.float32)

            # Reshape to 2D tensor (32x32)
            image = torch.from_numpy(image_data).float().view(1, self.image_size, self.image_size)

        elif self.mode == 'test':
            # test의 경우 image data만 뽑아낼 것임
            # Get image data
            image_data = self.df.iloc[idx, self.data_start_column - 1:].values.astype(np.float32)

            # Reshape to 2D tensor (32x32)
            image = torch.from_numpy(image_data).float().view(1, self.image_size, self.image_size)
            label_idx = torch.tensor(0)

        return image, label_idx

if __name__ == "__main__":
    train_csv_path = "./train.csv"
    test_csv_path = "./test.csv"

    train_dataset = ImageDataset(train_csv_path, mode = 'train')
    test_dataset = ImageDataset(test_csv_path, mode = 'test')

    print(f"label map : {train_dataset.label_map}")

    train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

    input, _ = next(iter(test_dataloader))

    print(f"test batch input shape: {input.shape}")

    image_data, _ = test_dataset[0]
    
    print(f"image data shape : {image_data.shape}")
    print(f"len test dataset : {len(test_dataset)}")
