import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Assume the dataset is in the following format:
# - Each image has corresponding annotation in a dictionary with keys 'boxes' and 'labels'
# - 'boxes' is a list of bounding boxes, each box is [xmin, ymin, xmax, ymax]
# - 'labels' is the class label of each box, for manhole detection, this can be a list of 1s


### Transforms ###
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # Augmentation
#     transforms.Resize((256, 256)),      # Preprocessing to resize images
#     transforms.ToTensor(),              # Preprocessing to convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Preprocessing for normalization
#                          std=[0.229, 0.224, 0.225])
# ])


class ManholeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        annotation = torch.load(ann_path)
        
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for num_classes (manhole + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def get_transform():
    # Define data transformations
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def main():
    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Our dataset has two classes only - background and manhole
    num_classes = 2
    dataset = ManholeDataset('path/to/your/dataset', get_transform())
    dataset_test = ManholeDataset('path/to/your/dataset', get_transform())
    
    # Split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    # Get the model using our helper function
    model = get_model(num_classes)
    
    # Move model to the right device
    model.to(device)
    
    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # And a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Let's train the model for 10 epochs
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Update the learning rate
        lr_scheduler.step()
        
        # Evaluate on the test dataset
        model.eval()
        with torch.no_grad():
            for images, targets in data_loader_test:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                prediction = model(images)
                
                # Evaluate predictions
                # You can add evaluation code here
        
        print(f"Epoch {epoch} completed")
    
    print("Training complete")

if __name__ == "__main__":
    main()
