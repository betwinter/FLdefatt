import torchvision as tv
class Dataset:
    def get_dataset(dir, name):
        if name == 'mnist':
            train_dataset = tv.datasets.MNIST(dir, train=True, transform=tv.transforms.ToTensor())
            eval_dataset = tv.datasets.MNIST(dir, train=False, transform=tv.transforms.ToTensor())
        elif name == 'cifar':
            transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4), tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = tv.datasets.CIFAR10(dir, train=True, transform=transform_train)

            eval_dataset = tv.datasets.CIFAR10(dir, train=False, transform=transform_test)
        return train_dataset, eval_dataset
