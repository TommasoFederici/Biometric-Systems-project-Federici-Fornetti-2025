import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1
from collections import Counter

# Configuration of dataset and constants for the model training
DATASET_PATH = r"PATH_TO_TRAINING_SET"  # Dataset already splitted in 80% train+validation and 20% test
IMG_SIZE = 160                          # FaceNet requires 160x160 images
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to map age to one of the 8 classes [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, ...)]
def age_to_class(age):
    '''
    Map the age to one of the 8 classes.
    :param age: age of the person
    :return: class of the age
    '''
    if 0 <= age <= 2: return 0
    elif 4 <= age <= 6: return 1
    elif 8 <= age <= 13: return 2
    elif 15 <= age <= 20: return 3
    elif 25 <= age <= 32: return 4
    elif 38 <= age <= 43: return 5
    elif 48 <= age <= 53: return 6
    elif age >= 60: return 7
    else: return -1  # Error class


# Function to apply Gaussian blur, CLAHE and Gabor filter to the image
def img_preprocessing(image, ksize=31, sigma=2.0, lambd=7.0, gamma=0.5, psi=0, orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4], multi_scale=True):
    '''
    Apply Gaussian blur, CLAHE and Gabor filter to the image.
    :param image: input image
    :param ksize: size of the Gabor kernel
    :param sigma: standard deviation of the Gaussian kernel
    :param lambd: wavelength of the sinusoidal factor
    :param gamma: spatial aspect ratio
    :param psi: phase offset
    :param orientations: list of orientations for the Gabor filter
    :param multi_scale: if True, apply the Gabor filter at two different scales
    :return: preprocessed image   
    '''
    
    # Gaussian blur for noise reduction
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    smoothed_image = clahe.apply(smoothed_image)

    # Gabor filter for texture analysis
    filtered_images = []
    for theta in orientations:
        for scale in [sigma, sigma*2]:  # Usa scale diverse per evidenziare dettagli fini e grossolani
            gabor_kernel = cv2.getGaborKernel((ksize, ksize), scale, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            filtered_image = cv2.filter2D(smoothed_image, cv2.CV_32F, gabor_kernel)
            filtered_images.append(filtered_image)

    # Combine all filtered images
    combined_filtered_image = np.sum(filtered_images, axis=0)

    # Normalize the combined image
    normalized_image = cv2.normalize(combined_filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_image = np.uint8(np.clip(normalized_image, 0, 255))

    return normalized_image


# Custom Dataset class for the Face Dataset
class FaceDataset(Dataset):
    def __init__(self, img_paths, genders, age_ranges, transform=None):
        self.img_paths = img_paths
        self.genders = genders
        self.age_ranges = age_ranges
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # preprocessing of the image to pass to the model
        img = img_preprocessing(img)  

        # FaceNet requires 3 channels, so we repeat the gray one to have 3 channels
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2) 

        # Apply the transformations and data augmentation
        if self.transform:
            img = self.transform(img)

        # Convert the labels to tensors
        gender = self.genders[idx]
        age_range = self.age_ranges[idx]

        return img, torch.tensor(gender, dtype=torch.long), torch.tensor(age_range, dtype=torch.long)

# Load the dataset and create the train and validation sets
img_paths = []
genders = []
age_ranges = []

for img_name in os.listdir(DATASET_PATH):
    try:
        age, gender, *_ = img_name.split('_')
        age = int(age)
        gender = int(gender)
        img_path = os.path.join(DATASET_PATH, img_name)

        img_paths.append(img_path)
        genders.append(gender)
        age_ranges.append(age_to_class(age))
    except Exception as e:
        continue

# Data splitting (train 70%, validation 10%) (already splitted in 80% train+validation and 20% test) 
train_paths, val_paths, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(
    img_paths, genders, age_ranges, test_size=0.125, random_state=42
)


# Data augmentation and transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),                          # Data augmentation
    transforms.RandomRotation(15),                              # Data augmentation
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the DataLoader for the train and validation sets
train_dataset = FaceDataset(train_paths, y_gender_train, y_age_train, transform=transform)
val_dataset = FaceDataset(val_paths, y_gender_val, y_age_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the FaceNet model pretrained on VGGFace2
base_model = InceptionResnetV1(pretrained='vggface2')

# Freeze the base model parameters
for param in base_model.parameters():
    param.requires_grad = False

# Unlock the last linear layer for fine-tuning
for param in base_model.last_linear.parameters():
    param.requires_grad = True


# Define the Multi-Task model
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.2)
        self.gender_head = nn.Linear(512, 2)  # gender output
        self.age_head = nn.Linear(512, 8)     # age output

    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)
        gender_output = self.gender_head(x)
        age_output = self.age_head(x)
        return gender_output, age_output

model = MultiTaskModel(base_model)

model.to(device)

# Weight for balancing the classes
# Compute the distribution of the age classes
age_range_counts = Counter(y_age_train)
total_age_samples = len(y_age_train)

# Compute the weights for each class
age_weights = {age: total_age_samples / count for age, count in age_range_counts.items()}
age_class_weights = torch.tensor([age_weights[i] for i in range(8)], dtype=torch.float32).to(device)

# Define the loss function and the optimizer
criterion_gender = nn.CrossEntropyLoss()
criterion_age = nn.CrossEntropyLoss(weight=age_class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training function
def train_epoch(model, dataloader, optimizer, criterion_gender, criterion_age):
    model.train()
    total_loss = 0
    correct_gender = 0
    correct_age = 0
    total_samples = 0

    for images, genders, ages in dataloader:
        images, genders, ages = images.to(device), genders.to(device), ages.to(device)

        optimizer.zero_grad()
        gender_preds, age_preds = model(images)

        loss_gender = criterion_gender(gender_preds, genders)
        loss_age = criterion_age(age_preds, ages)
        loss = loss_gender + loss_age

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_gender += (gender_preds.argmax(1) == genders).sum().item()
        correct_age += (age_preds.argmax(1) == ages).sum().item()
        total_samples += genders.size(0)

    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total_samples
    age_acc = correct_age / total_samples

    return avg_loss, gender_acc, age_acc

# Evaluation function
def evaluate(model, dataloader, criterion_gender, criterion_age):
    model.eval()
    total_loss = 0
    correct_gender = 0
    correct_age = 0
    total_samples = 0

    with torch.no_grad():
        for images, genders, ages in dataloader:
            images, genders, ages = images.to(device), genders.to(device), ages.to(device)

            gender_preds, age_preds = model(images)

            loss_gender = criterion_gender(gender_preds, genders)
            loss_age = criterion_age(age_preds, ages)
            loss = loss_gender + loss_age

            total_loss += loss.item()
            correct_gender += (gender_preds.argmax(1) == genders).sum().item()
            correct_age += (age_preds.argmax(1) == ages).sum().item()
            total_samples += genders.size(0)

    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total_samples
    age_acc = correct_age / total_samples

    return avg_loss, gender_acc, age_acc


# Patience parameter for Early Stopping
patience = 5                        # Number of epochs without improvement before stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

# Training loop with Early Stopping and Learning Rate Scheduler
for epoch in range(EPOCHS):
    train_loss, train_gender_acc, train_age_acc = train_epoch(
        model, train_loader, optimizer, criterion_gender, criterion_age
    )
    val_loss, val_gender_acc, val_age_acc = evaluate(
        model, val_loader, criterion_gender, criterion_age
    )

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Gender Acc: {train_gender_acc:.4f}, Train Age Acc: {train_age_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Gender Acc: {val_gender_acc:.4f}, Validation Age Acc: {val_age_acc:.4f}")

    # Save the model if the validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')  
        epochs_without_improvement = 0  # Reset the counter
    else:
        epochs_without_improvement += 1

    # Early Stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping after {epoch+1} epochs.")
        break

    # Learning Rate Scheduler
    scheduler.step(val_loss)
