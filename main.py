import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from Net import Net
from FeatureExtraction import FeatureExtraction
import matplotlib.pyplot as plt

def get_data_loader(is_train = False):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST('', is_train, transform=to_tensor, download=True) # 用于下载数据集，字符串空表示下载到当前目录，is_train表示下载训练集还是测试集
    return DataLoader(data_set, batch_size=15, shuffle=True) # shuffle表示数据是否是随机打乱的


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


if __name__ == '__main__':
    train_data = get_data_loader(True)
    test_data = get_data_loader()
    f = FeatureExtraction()
    net = Net(3, 64, 10)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(20):
        for (n, (x, y)) in enumerate(train_data):
            net.zero_grad()
            output = net.forward(f.get_feature(x))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for (x, y) in test_data:
                outputs = net.forward(f.get_feature(x))
                for i, output in enumerate(outputs):
                    if torch.argmax(output) == y[i]:
                        n_correct += 1
                    n_total += 1
        print("epoch", epoch, "accuracy:", n_correct / n_total)

    torch.save(net, 'model.pth')
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        outputs = net.forward(f.get_feature(x))
        predict = torch.argmax(outputs[0])
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


