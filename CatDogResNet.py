import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import visdom
import os

#vis = visdom.Visdom()
#vis.close(env="main")

# 推理函数：使用训练好的模型预测用户自己的图片
def predict_image(image_path):
    """
    使用训练好的模型预测单张图片
    :param image_path: 图片路径
    :return: 预测结果
    """
    # 导入resnet模块
    import resnet
    # 加载模型
    model = resnet.resnet50(num_classes=2, zero_init_residual=True).to(device)

    # 查找最新保存的模型权重
    import glob
    model_files = glob.glob('./model/*.pth')
    if not model_files:
        print('No model weights found!')
        return

    # 按修改时间排序，选择最新的模型
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    print(f'Loading model: {latest_model}')

    # 加载模型权重
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()  # 设置为评估模式

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert('RGB')),  # 将RGBA转换为RGB
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图片
    from PIL import Image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度
    image = image.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # 定义类别
    classes = ('cats', 'dogs')
    # 输出结果
    class_name = classes[predicted.item()]
    print(f'Prediction: {class_name}')
    return class_name

#def value_tracker(value_plot, value, num):
    '''num, loss_value, are Tensor'''
    #vis.line(X=num,
    #         Y=value,
    #         win=value_plot,
    #         update='append'
    #         )


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

if __name__ == '__main__':
    import sys
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # 推理模式
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
            predict_image(image_path)
        else:
            print('Usage: python CatDogResNet.py predict <image_path>')
    else:
        # 训练模式（默认）
        print('Starting training...')

        # CatAndDog数据集的转换
        transform_train = transforms.Compose([
            transforms.Resize(256),  # 先将最小边长调整为256
            transforms.RandomCrop(224),  # 随机裁剪为224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准归一化
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),  # 先将最小边长调整为256
            transforms.CenterCrop(224),  # 中心裁剪为224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准归一化
        ])

        # 加载CatAndDog数据集
        trainset = torchvision.datasets.ImageFolder(root='CatAndDog/training_set/training_set', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.ImageFolder(root='CatAndDog/test_set/test_set', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=0)

        # CatAndDog的类别
        classes = ('cats', 'dogs')

        import resnet

        # 使用修改后的ResNet50模型，输出类别数为2（CatAndDog的二分类任务）
        resnet50 = resnet.resnet50(num_classes=2, zero_init_residual=True).to(device)
        # 1(conv1) + 9(layer1) + 12(layer2) + 18(layer3) + 9(layer4) +1(fc)= ResNet50

        # 创建model目录（如果不存在）
        if not os.path.exists('./model'):
            os.makedirs('./model')

        resnet50

        a = torch.Tensor(1, 3, 224, 224).to(device)
        out = resnet50(a)
        print(out)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001, weight_decay=5e-4)
        lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        #loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
        #acc_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))

        def acc_check(net, test_set, epoch, save=1):
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_set:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = (100 * correct / total)
            print('Accuracy of the network on the test images: %.2f %%' % acc)
            if save:
                torch.save(net.state_dict(), "./model/model_epoch_{}_acc_{:.2f}.pth".format(epoch, acc))
            return acc

        print(len(trainloader))
        epochs = 30

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            resnet50.train()  # 设置模型为训练模式
            lr_sche.step()
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = resnet50(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    #value_tracker(loss_plt, torch.Tensor([running_loss / 100]), torch.Tensor([i + epoch * len(trainloader)]))
                    print('[%d, %5d] loss: %.3f, lr: %.6f' %
                          (epoch + 1, i + 1, running_loss / 100, optimizer.param_groups[0]['lr']))
                    running_loss = 0.0

            # Check Accuracy
            resnet50.eval()  # 设置模型为评估模式
            acc = acc_check(resnet50, testloader, epoch, save=1)
            #value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))

        print('Finished Training')

        # 最终测试
        correct = 0
        total = 0
        resnet50.eval()  # 设置模型为评估模式
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = resnet50(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Final accuracy of the network on the test images: %.2f %%' % (
                100 * correct / total))


