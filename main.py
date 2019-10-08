from time import sleep
import numpy as np
from PQP import PQP

def test_CIFAR10():
    print('test PQP attack for a black-box trained on ImageNet')
    import torch
    import torchvision
    from torchvision import transforms
    import resnet

    print('*** defining the black box to attack ***')
    def get_blackbox(model_name):
        if model_name.lower() in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
            model = torch.nn.DataParallel(resnet.__dict__[model_name.lower()]())
            model.cuda()
            checkpoint = torch.load('CIFAR10_pretrained_models/' + model_name + '.th')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError(model_name, 'Input network name not recognized.')
        model.eval()
        return model

    model = get_blackbox('resnet32')

    print('*** defining data loader and data preprocessing ***')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./CIFAR10_dataset/', train=False, transform=transforms.ToTensor(), download=True),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    data_loader = ((img.numpy().squeeze().transpose(1,2,0) * 255., label.numpy().squeeze())
                   for img, label in test_loader)

    print('*** defining the python function implementing the query to the black box ***')
    def forward(img):
        '''
        :param img: must be a 3D channel-last np array in the range [0, 255]
        :return: numpy array (either the softmax output probabilities or the extracted feature vector)
        '''
        with torch.no_grad():
            img = transform(img.astype(np.uint8)).unsqueeze(0).cuda()
            output = torch.nn.Softmax(dim=1)(model(img)).squeeze()
            output = output.data.cpu().numpy()
        return output

    attack_classifier(forward, data_loader, hard_attack=True)

def attack_classifier(forward, data_generator, num_imgs=100, hard_attack=True, loss_goal=0.9, N=20):
    query_fun = lambda img, target: forward(img)[target]

    mean = lambda x: np.asarray(x).mean()
    success, ssim, psnr, NQ = [], [], [], []
    for img, label in data_generator:
        # select the target class
        probs = forward(img)
        if loss_goal is None:
            loss_goal_ = np.max(probs)
        else:
            loss_goal_ = loss_goal
        preds = np.argsort(probs)
        if preds[-1] != label:
            continue  # skip black-box error
        if hard_attack:
            target = preds[0]  # last predicted class
            print('\n\n\n*** attacking image labelled %d with target label %d (hard case) ***\n\n\n' % (label, target))
            print_every = 100
        else:
            target = preds[-2] # 2nd predicted class
            print('\n\n\n*** attacking image labelled %d with target label %d (easy case) ***\n\n\n' % (label, target))
            print_every = 10

        # start attack
        _, success_, ssim_, psnr_, NQ_, _ = PQP(query_fun=query_fun, or_img=img, target=target, loss_goal=loss_goal_, N=N,
                                                minimize_loss=False, print_every=print_every)
        success.append(success_)
        ssim.append(ssim_)
        psnr.append(psnr_)
        NQ.append(NQ_)
        print('\n*** samples: %d/%d   Avg. metrics: succ. rate %0.3f   ssim: %0.3f   psnr: %0.3f   NQ: %0.0f'
              % (len(success), num_imgs, mean(success), mean(ssim), mean(psnr), mean(NQ)))
        if len(success) == num_imgs:
            break
        else:
            sleep(3)

if __name__ == "__main__":
    test_CIFAR10()