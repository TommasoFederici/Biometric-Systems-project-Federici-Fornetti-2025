import numpy as np
import os
import cv2
import torch
import torch.nn as nn  
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


IMG_SIZE = 160  
MODEL_PATH = 'best_model.pth'  
DATASET_PATH = r"PATH_TO_TEST_SET"  
NUM_CLASSES = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    smoothed_image = clahe.apply(smoothed_image)

    filtered_images = []
    for theta in orientations:
        for scale in [sigma, sigma*2]:
            gabor_kernel = cv2.getGaborKernel((ksize, ksize), scale, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            filtered_image = cv2.filter2D(smoothed_image, cv2.CV_32F, gabor_kernel)
            filtered_images.append(filtered_image)
    combined_filtered_image = np.sum(filtered_images, axis=0)

    normalized_image = cv2.normalize(combined_filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_image = np.uint8(np.clip(normalized_image, 0, 255))

    return normalized_image

# Load of the pretrained multi-task model
base_model = InceptionResnetV1(pretrained='vggface2').to(device)  # Carica anche su GPU

class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.2)
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, 8)   

    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)
        gender_output = self.gender_head(x)
        age_output = self.age_head(x)
        return gender_output, age_output

model = MultiTaskModel(base_model)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

model.to(device)

# Load the images and labels form the test set folder
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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

total_samples = len(img_paths)

# Variables for Miss Rate (or FNR), Fall-Out (or FPR), Accuracy, Precision and Recall for Gender
correct_gender = 0
tp_male = 0
fn_male = 0
fp_male = 0
tn_male = 0

# For the ROC curve
predicted_probs = []

# Variables for Age 
correct_age = 0
tp = [0] * NUM_CLASSES
tn = [0] * NUM_CLASSES
fp = [0] * NUM_CLASSES
fn = [0] * NUM_CLASSES

fnr = [0] * NUM_CLASSES
fpr = [0] * NUM_CLASSES
precision = [0] * NUM_CLASSES
recall = [0] * NUM_CLASSES
F_score = [0] * NUM_CLASSES

error_distances = []

Confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))


# Loop over the test set
with torch.no_grad():
    for i, img_path in enumerate(img_paths):
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # preprocessing
        img = img_preprocessing(img)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = transform(img).unsqueeze(0).to(device)  
        
        # Real labels
        true_gender = genders[i]
        true_age_range = age_ranges[i]
        
        # Prediction
        pred_gender_logits, pred_age_logits = model(img)

        # Softmax for probabilities computation
        probs = F.softmax(pred_gender_logits, dim=1)
        predicted_prob = probs[:, 1].cpu().numpy()
        predicted_probs.append(predicted_prob)
        
        pred_gender = torch.argmax(pred_gender_logits, dim=1).item()
        pred_age_range = torch.argmax(pred_age_logits, dim=1).item()
        
        # Comparison with real labels (gender)
        if pred_gender == true_gender:
            if true_gender == 0:
                tp_male += 1
            else:
                tn_male += 1
        else:
            if true_gender == 0:
                fn_male += 1
            else:
                fp_male += 1

        # Comparison with real labels (age)
        if pred_age_range == true_age_range:
            correct_age += 1
            tp[true_age_range] += 1
            for i in range(NUM_CLASSES):
                if i != true_age_range:
                    tn[i] += 1
            error_distances.append(0)
            Confusion_matrix[true_age_range][true_age_range] += 1
        else:
            fp[true_age_range] += 1
            fn[pred_age_range] += 1
            for i in range(NUM_CLASSES):
                if i != true_age_range and i != pred_age_range:
                    tn[i] += 1
            error_distances.append(abs(true_age_range - pred_age_range))
            Confusion_matrix[true_age_range][pred_age_range] += 1    

# Miss Rate (FNR) and Fall-Out (FPR) computation
FNR_male = fn_male/(tp_male + fn_male)
FPR_male = fp_male/(fp_male + tn_male)

# Accuracy, Precision, Recall, F-score computation for Gender
correct_gender = tp_male + tn_male
gender_accuracy = correct_gender / total_samples
precision_male = tp_male/(tp_male + fp_male)
recall_male = tp_male/(tp_male + fn_male)
precision_female = tn_male/(tn_male + fn_male)
recall_female = tn_male/(tn_male + fp_male)
F_score_male = 2 * (precision_male * recall_male) / (precision_male + recall_male)
F_score_female = 2 * (precision_female * recall_female) / (precision_female + recall_female)


print(f"Accuracy gender: {gender_accuracy:.2%}")

print(f"Miss Rate (FNR) male: {FNR_male:.2%}")
print(f"Fall-Out (FPR) female: {FPR_male:.2%}")

print(f"Precision male: {precision_male:.2%}")
print(f"Recall male: {recall_male:.2%}")
print(f"Precision female: {precision_female:.2%}")
print(f"Recall female: {recall_female:.2%}")
print(f"F-SCORE male: {F_score_male:.2%}")
print(f"F-SCORE female: {F_score_female:.2%}")

######################################################

# FNR, FPR, Accuracy, Precision, Recall, F-score computation for Age
age_accuracy = correct_age / total_samples
for i in range(NUM_CLASSES):
            fnr[i] = round((fn[i] / (tp[i] + fn[i]) * 100), 2) if (tp[i] + fn[i]) > 0 else 0
            fpr[i] = round((fp[i] / (fp[i] + tn[i]) * 100), 2) if (fp[i] + tn[i]) > 0 else 0
            precision[i] = round((tp[i] / (tp[i] + fp[i]) * 100), 2) if (tp[i] + fp[i]) > 0 else 0
            recall[i] = round((tp[i] / (tp[i] + fn[i]) * 100), 2) if (tp[i] + fn[i]) > 0 else 0
            F_score[i] = round((2 * (precision[i] * recall[i]) / (precision[i] + recall[i])), 2) if (precision[i] + recall[i]) > 0 else 0

# Error distance mean and standard deviation
mean_error = np.mean(error_distances)
std_error = np.std(error_distances)

print("-----------------------------------\n")
print(f"Accuracy age: {age_accuracy:.2%}")

print(f"FNR age: {fnr}")                            # They are printed as lists
print(f"FPR age: {fpr}")
print(f"precision age: {precision}")
print(f"recall age: {recall}")
print(f"f-score age: {F_score}")
print("\n")
print(f"Mean error: {mean_error:.2f}")
print(f"Standard deviation error: {std_error:.2f}")

#######################################################
# ROC curve
fpr, tpr, thresholds = roc_curve(genders, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR = 1-FNR')
plt.title('Gender ROC Curve (first system)')
plt.legend(loc='lower right')
plt.show()

#######################################################
# Plot of error distances
plt.figure(figsize=(8,5))
plt.hist(error_distances, bins=range(0, max(error_distances)+2), alpha=0.7, color='blue', edgecolor='black')
plt.xticks(range(0, max(error_distances)+1))
plt.xlabel("Error distance")
plt.ylabel("Frequency")
plt.title("Distribution of error distances on age prediction (first system)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#######################################################
# Confusion matrix in percentage
normalized_cm = (Confusion_matrix.T / Confusion_matrix.sum(axis=1)).T * 100

# Plot della matrice di confusione in percentuale
plt.figure(figsize=(8, 6))
plt.imshow(normalized_cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix Age Prediction Architecture 1")
plt.colorbar(format="%.2f%%")
plt.xticks(range(NUM_CLASSES))
plt.yticks(range(NUM_CLASSES))
plt.xlabel("Predicted class")
plt.ylabel("Real class")

# Add the percentage values
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, f"{normalized_cm[i, j]:.2f}%", 
                 ha='center', va='center', color='black')
plt.show()

