from torchvision.datasets import CIFAR10

def prepare_data():
    train_dataset = CIFAR10("CIFAR10/train", download=True)
    test_dataset = CIFAR10("CIFAR10/test", download=True)

    return train_dataset, test_dataset

if __name__ == "__main__":
    prepare_data()
