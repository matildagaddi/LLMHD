import torch
import torchhd
from loader import EMNLP18
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The norm of a vector is nearly zero, this could indicate a bug.')

classifiers = [
    "Vanilla",
    "AdaptHD",
    "OnlineHD",
    "NeuralHD",
    "DistHD",
    "CompHD",
    "SparseHD",
    "QuantHD",
    "LeHDC",
    # "IntRVFL",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 16384  # number of hypervector dimensions # must be power of 2 for compHD
BATCH_SIZE = 12  # for GPUs with enough memory we can process multiple images at ones
EPOCHS = 70
CLASSTYPE = 'fact' #fact or bias

train_ds = EMNLP18(root="", train=True, classification=CLASSTYPE)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = EMNLP18(root="", train=False, classification=CLASSTYPE)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

num_features = train_ds[0][0].size(-1)
num_classes = len(train_ds.classes)

std, mean = torch.std_mean(train_ds.data, dim=0, keepdim=False)


def transform(sample):
    return (sample - mean) / std


train_ds.transform = transform
test_ds.transform = transform

params = {
    "Vanilla": {},
    "AdaptHD": {
        "epochs": EPOCHS,
    },
    "OnlineHD": {
        "epochs": EPOCHS,
    },
    "NeuralHD": {
        "epochs": EPOCHS,
        "regen_freq": 5,
    },
    "DistHD": {
        "epochs": EPOCHS,
        "regen_freq": 5,
    },
    "CompHD": {},
    "SparseHD": {
        "epochs": EPOCHS,
    },
    "QuantHD": {
        "epochs": EPOCHS,
    },
    "LeHDC": {
        "epochs": EPOCHS,
    },
    "IntRVFL": {}, #killed. There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '
}

for classifier in classifiers:
    print()
    print(classifier, EPOCHS, CLASSTYPE)
    
    # Initialize the model
    model_cls = getattr(torchhd.classifiers, classifier)
    model: torchhd.classifiers.Classifier = model_cls(
        num_features, DIMENSIONS, num_classes, device=device, **params[classifier]
    )

    # Fit the model
    model.fit(train_ld)
    accuracy = model.accuracy(test_ld)
    print(f"Testing accuracy of {(accuracy * 100):.3f}%")

    # Collect predictions and true labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_ld:
            inputs, labels = batch
            outputs = model.predict(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print confusion matrix
    print(f"Confusion Matrix for {classifier}, epochs: {EPOCHS}, {CLASSTYPE}:")
    print(cm)

    # # Optional: Plot confusion matrix
    # fig = plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_ds.classes, yticklabels=train_ds.classes)
    # plt.title(f'Confusion Matrix for {classifier}')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # fig_lst.append(fig)

    # Optional: Print classification report
    print(f"Classification Report for {classifier}:")
    print(classification_report(y_true, y_pred, target_names=train_ds.classes))
