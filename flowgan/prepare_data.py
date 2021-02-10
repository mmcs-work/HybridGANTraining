import torchvision
import torch
import torchvision.transforms as transforms


def get_data(opt):
    if opt.dataset == 'mnist':
        
        train = torchvision.datasets.MNIST(root=opt.root_dir, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
#                                transforms.Normalize((0.5,), (0.5,)),
                           ]))
        
        test = torchvision.datasets.MNIST(root=opt.root_dir, download=True,train = False,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.img_size),
                                   transforms.ToTensor(),
#                                    transforms.Normalize((0.5,), (0.5,)),
                               ]))
        train_dataset, val_dataset = torch.utils.data.random_split(train, [50000,10000])
#         train_dataset, val_dataset, _ = torch.utils.data.random_split(train, [100, 100, 59800])  # for debug
        
        
    
    elif opt.dataset == 'cifar10':
        
        train = torchvision.datasets.CIFAR10(root=opt.root_dir, 
                                             train=True, 
                                             transform=transforms.Compose([
                                                       transforms.Resize(opt.img_size),
                                                       transforms.ToTensor(),
#                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                       ]), 
                                             download=True)
        
        test = torchvision.datasets.CIFAR10(root=opt.root_dir, 
                                             train=False, 
                                             transform=transforms.Compose([
                                                       transforms.Resize(opt.img_size),
                                                       transforms.ToTensor(),
#                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                       ]), 
                                             download=True)
        
        train_dataset, val_dataset = torch.utils.data.random_split(train, [40000,10000])
#         train_dataset, val_dataset, _ = torch.utils.data.random_split(train, [100, 100, 49800])  # for debug
    
    elif opt.dataset == 'fmnist':
        
        train = torchvision.datasets.FashionMNIST(root=opt.root_dir, 
                                                  train=True, transform=transforms.Compose([
                                                       transforms.Resize(opt.img_size),
                                                       transforms.ToTensor(),
#                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                       ]), 
                                                  download=True)
        
        test = torchvision.datasets.FashionMNIST(root=opt.root_dir, 
                                                  train=False, transform=transforms.Compose([
                                                       transforms.Resize(opt.img_size),
                                                       transforms.ToTensor(),
#                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                       ]), 
                                                  download=True)
        
        train_dataset, val_dataset = torch.utils.data.random_split(train, [50000,10000])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=2)
    
    return train_dataloader, val_dataloader, test_dataloader