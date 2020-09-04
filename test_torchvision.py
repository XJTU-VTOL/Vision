from torchvision.datasets import CocoDetection
import torchvision.transforms as T 

trans = T.Compose(
    [T.ToTensor(),
    T.RandomHorizontalFlip(0.5)]
)

dataset = CocoDetection(root='/data/cxg1/Data/train2014', annFile='/data/cxg1/Data/annotations/instances_train2014.json')
print('Number of samples: ', len(dataset))

print(type(dataset))

images, label = dataset[3]
pass