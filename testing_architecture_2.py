import cv2 as cv
import os
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Paths to model configuration and weights
ageProto = r"models_architecture2\age_deploy.prototxt"
ageModel = r"models_architecture2\age_net.caffemodel"

genderProto = r"models_architecture2\gender_deploy.prototxt"
genderModel = r"models_architecture2\gender_net.caffemodel"

# Folder containing test images
dataset_folder = r"PATH_TO_TEST_SET"

# Mean values used for model preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender categories
ageList = ['(0-2)','(4-6)', '(8-13)', '(15-20)', '(25-35)','(38-44)','(48-53)','(60-100)']
genderList = ['Male', 'Female']
NUM_CLASSES = 8

# Load the pre-treied models
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

def age_gender_detector(face):
    """
    Detects age and gender from a given face image.
    :param face: Input image (200x200)
    :return: Predicted age range and gender
    """

    if face is None:
        print("Errore: immagine non caricata correttamente.")
        return None

    if face.shape[0] != 200 or face.shape[1] != 200:
        print(f"Errore: l'immagine deve essere 200x200, ma è {face.shape}.")
        return None

    # Preprocess the image for the neural network
    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age, gender

def map_age_to_range(age):
    """
    Maps a numerical age to a predefined age range.
    :param age: Integer age
    :return: Age range category as a string
    """
    if age <= 2:
        return '(0-2)'
    elif age <= 6 and age>= 4:
        return '(4-6)'
    elif age <= 13 and age>= 8:
        return '(8-13)'
    elif age <= 20 and age>= 15:
        return '(15-20)'
    elif age <= 35 and age>= 25:
        return '(25-35)'
    elif age <= 44 and age>= 38:
        return '(38-44)'
    elif age <= 53 and age>= 48:
        return '(48-53)'
    elif age >= 60:
        return '(60-100)'
    
def map_range_to_class(age_range):
    mapping = {
        '(0-2)': 0, '(4-6)': 1, '(8-13)': 2, '(15-20)': 3,
        '(25-35)': 4, '(38-44)': 5, '(48-53)': 6, '(60-100)': 7
    }
    return mapping.get(age_range, -1)  # -1 per identificare errori


def create_dataset_array(dataset_folder):
    """
    Creates an array of ground truth age and gender labels from dataset filenames.
    :param dataset_folder: Folder containing images named as 'age_gender.jpg'
    :return: List of tuples (age range, gender)
    """
    dataset_array = []
    image_files = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder)]  # Percorsi assoluti

    for image in image_files:
        parts = os.path.basename(image).split('_')  # Estrai il nome del file
        age = int(parts[0])
        range_age = map_age_to_range(age)
        gender = 'Female' if int(parts[1]) == 1 else 'Male'
        dataset_array.append((range_age, gender))

    return dataset_array

def evaluation(dataset_folder):
    """
    Evaluates the performance of the age and gender detection system.
    :param dataset_folder: Folder containing test images
    :return: List of age prediction error distances
    """
    perc = 0
    cont = 0

    age_acc = 0
    gender_acc = 0
    correct_gender = 0
    correct_age = 0
    
    tp_male = 0
    fn_male = 0
    fp_male = 0
    tn_male = 0

    result_array = []

    tp = [0] * len(ageList)
    tn = [0] * len(ageList)
    fp = [0] * len(ageList)
    fn = [0] * len(ageList)

    FNR_age = [0] * len(ageList)
    FPR_age = [0] * len(ageList)
    precision_age = [0] * len(ageList)
    recall_age = [0] * len(ageList)
    F_score_age = [0] * len(ageList)

    error_distances= []
    Confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    #for roc curve
    true_gender_labels = []  
    gender_scores = [] 

    dataset_array = create_dataset_array(dataset_folder)

    image_files = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder)]
    print(f"Analisi al 0%")

    total_samples = len(image_files)

    for image in image_files:
        input_img = cv.imread(image)

        if input_img is None:
            print(f"Errore: impossibile aprire {image}")
            continue

        (pred_age_range, pred_gender) = age_gender_detector(input_img)
        (true_age_range, true_gender) = dataset_array[cont]
        
        ###
        percentuale_analisi = int((cont / total_samples) * 100)
        if percentuale_analisi != perc:  # Stampa solo se cambia
            print(f"{percentuale_analisi}%")
            perc = percentuale_analisi  # Aggiorna perc con il nuovo val
        cont += 1
        ##

        genderPreds = genderNet.forward()
        true_gender_labels.append(0 if true_gender == 'Male' else 1)  # 1 per Male, 0 per Female
        gender_scores.append(genderPreds[0][1])
        ##

        #comparison of the real label with the predicted one
        result_array.append(pred_age_range)
        if true_age_range == pred_age_range:
            correct_age += 1

            true_age_class = map_range_to_class(true_age_range)
            pred_age_class = map_range_to_class(pred_age_range)
            
            tp[true_age_class] += 1
            for i in range(len(ageList)):
                if i !=true_age_class:
                    tn[i] += 1
            error_distances.append(abs(pred_age_class - true_age_class))
            Confusion_matrix[true_age_class][true_age_class] += 1

        elif true_age_range != pred_age_range:
            true_age_class = map_range_to_class(true_age_range)
            pred_age_class = map_range_to_class(pred_age_range)
            fp[true_age_class] += 1
            fn[pred_age_class] += 1
            for i in range(len(ageList)):
                if i != true_age_class and i != pred_age_class:
                    tn[i] += 1
            error_distances.append(abs(pred_age_class - true_age_class))
            Confusion_matrix[true_age_class][pred_age_class] += 1

        if true_gender == pred_gender:
            correct_gender += 1

            if true_gender == 'Male':
                tp_male += 1
            else:
                tn_male += 1
            
        elif true_gender != pred_gender:
            if true_gender == 'Male':
                fn_male += 1
            else:
                fp_male += 1


    #FNR FPR PRECISION RECALL F-SCORE (AGE)
    for i in range(len(ageList)):
        FNR_age[i] = round((fn[i] / (tp[i] + fn[i]) * 100), 2) if (tp[i] + fn[i]) > 0 else 0
        FPR_age[i] = round((fp[i] / (fp[i] + tn[i]) * 100), 2) if (fp[i] + tn[i]) > 0 else 0
        precision_age[i] = round((tp[i] / (tp[i] + fp[i]) * 100), 2) if (tp[i] + fp[i]) > 0 else 0
        recall_age[i] = round((tp[i] / (tp[i] + fn[i]) * 100), 2) if (tp[i] + fn[i]) > 0 else 0
        F_score_age[i] = round((2 * (precision_age[i] * recall_age[i]) / (precision_age[i] + recall_age[i])), 2) if (precision_age[i] + recall_age[i]) > 0 else 0
    
    age_acc = correct_age / total_samples
    gender_acc = correct_gender / total_samples
    ###

    #FNR FPR PRECISION RECALL F-SCORE (GENDER)
    FNR_male = fn_male/(tp_male + fn_male) if (tp_male + fn_male) > 0 else 0
    FPR_male = fp_male/(fp_male + tn_male) if (fp_male + tn_male) > 0 else 0

    precision_male = tp_male/(tp_male + fp_male) if (tp_male + fp_male) > 0 else 0
    recall_male = tp_male/(tp_male + fn_male) if (tp_male + fn_male) > 0 else 0 
    precision_female = tn_male/(tn_male + fn_male) if (tn_male + fn_male) > 0 else 0
    recall_female = tn_male/(tn_male + fp_male) if (tn_male + fp_male) > 0 else 0
    F_score_male = 2 * (precision_male * recall_male) / (precision_male + recall_male) if (precision_male + recall_male) > 0 else 0 
    F_score_female = 2 * (precision_female * recall_female) / (precision_female + recall_female) if (precision_female + recall_female) > 0 else 0
    ###

    mean_error = np.mean(error_distances)
    std_error = np.std(error_distances)

    #print of the result
    print("RESULTS:")
    print(f"Age accuracy: {age_acc:.2%}, Gender accuracy: {gender_acc:.2%}")

    print(f"FNR_male: {FNR_male:.2%}")
    print(f"FPR_male: {FPR_male:.2%}")

    print(f"Precision_male: {precision_male:.2%}")
    print(f"Recall_male: {recall_male:.2%}")
    print(f"Precision_female: {precision_female:.2%}")
    print(f"Recall_female: {recall_female:.2%}")
    print(f"F-SCORE_male: {F_score_male:.2%}")
    print(f"F-SCORE_female: {F_score_female:.2%}")

    print(f"FNR_age: {FNR_age}")
    print(f"FPR_age: {FPR_age}")
    print(f"Precision_age: {precision_age}")
    print(f"Recall_age: {recall_age}")
    print(f"F-SCORE_age: {F_score_age}")

    print(f"Mean Error: {mean_error:.2f}")
    print(f"Standard deviation: {std_error:.2f}")

    # for roc curve
    fpr, tpr, thresholds = roc_curve(true_gender_labels, gender_scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC (ROC) for gender: {roc_auc:.2f}")

    # Visualization of ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR = 1-FNR')
    plt.title('Gender ROC Curve (second system)')
    plt.legend(loc='lower right')
    plt.show()

    return error_distances, result_array, Confusion_matrix

def show_random_images(dataset_folder, num_images=6):
    """
    Displays random images from the dataset with their predicted and real labels.
    :param dataset_folder: Folder containing test images
    :param num_images: Number of images to display
    """
    image_files = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder)]
    random_images = random.sample(image_files, min(num_images, len(image_files)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, image_path in enumerate(random_images):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        parts = os.path.basename(image_path).split('_')
        true_age = int(parts[0])
        true_age_range = map_age_to_range(true_age)
        true_gender = 'Female' if int(parts[1]) == 1 else 'Male'
        
        pred_age_range, pred_gender = age_gender_detector(image)
        
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f"Real: {true_age_range}, {true_gender}\nPred: {pred_age_range}, {pred_gender}")
    
    plt.tight_layout()
    plt.show()

def error_distances_visualize(error_distances):
    """
    Graphically displays the distance of errors in the age prediction
    :Error_distances: Array containing the distance error of each image
    """
    plt.figure(figsize=(8,5))
    plt.hist(error_distances, bins=range(0, max(error_distances)+2), alpha=0.7, color='blue', edgecolor='black')
    plt.xticks(range(0, max(error_distances)+1))
    plt.xlabel("Error distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of error distances on age prediction (second system)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

error_distances, pred_labels, Confusion_matrix = evaluation(dataset_folder)

show_random_images(dataset_folder)
error_distances_visualize(error_distances)

#####################################
# Plot confusion matrix in percentage
normalized_cm = (Confusion_matrix.T / Confusion_matrix.sum(axis=1)).T * 100
plt.figure(figsize=(8, 6))
plt.imshow(normalized_cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix Age Prediction Architecture 2")
plt.colorbar(format="%.2f%%")
plt.xticks(range(NUM_CLASSES))
plt.yticks(range(NUM_CLASSES))
plt.xlabel("Predicted class")
plt.ylabel("Real class")

# Add percentages to the plot
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, f"{normalized_cm[i, j]:.2f}%", 
                 ha='center', va='center', color='black')
plt.show()