import torch


class Net(torch.nn.Module):
    def __init__(self, in_features_nums, mid_features_nums, out_features_nums):
        super().__init__()
        self.in_nums = in_features_nums
        self.mid_nums = mid_features_nums
        self.out_nums = out_features_nums
        self.fc1 = torch.nn.Linear(in_features_nums, mid_features_nums)
        self.fc2 = torch.nn.Linear(mid_features_nums, mid_features_nums)
        self.fc3 = torch.nn.Linear(mid_features_nums, mid_features_nums)
        self.fc4 = torch.nn.Linear(mid_features_nums, out_features_nums)

    def forward(self, x):
        x = x.view(-1, self.in_nums).to(torch.float32)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x