import numpy as np
import torch


class FeatureExtraction:

    def __get_one_zero_rate(self, image_tensor):
        image_tensor_clone = image_tensor.clone()
        if image_tensor_clone.dim() == 3:
            image_tensor_clone = image_tensor_clone[0]
        image_tensor_clone[image_tensor_clone > 0] = 1
        one_count = (image_tensor_clone == 1).sum().item()
        zero_count = (image_tensor_clone == 0).sum().item()
        return one_count / zero_count

    def __get_height_width_rate(self, image_tensor):
        image_tensor_clone = image_tensor.clone()
        if image_tensor_clone.dim() == 3:
            image_tensor_clone = image_tensor_clone[0]
        image_tensor_clone[image_tensor_clone > 0] = 1
        image_array = image_tensor_clone.numpy()
        nz_rows, nz_cols = np.nonzero(image_array)
        if len(nz_rows) == 0:
            return 0
        left, right = nz_cols.min(), nz_cols.max()
        top, bottom = nz_rows.min(), nz_rows.max()
        width = right - left + 1
        height = bottom - top + 1
        rate = height / width
        return rate


    def __get_top_bottom_nums_rate(self, image_tensor):
        image_tensor_clone = image_tensor.clone()
        if image_tensor_clone.dim() == 3:
            image_tensor_clone = image_tensor_clone[0]
        image_tensor_clone[image_tensor_clone > 0] = 1
        half_height = image_tensor_clone.size(0) // 2
        top_part = image_tensor_clone[:half_height]
        bottom_part = image_tensor_clone[half_height:]
        count_top = torch.sum(top_part == 1)
        count_bottom = torch.sum(bottom_part == 1)
        return count_top / count_bottom


    def get_feature(self, image_batch_tensor):
        features_tensor_list = []
        image_batch_tensor_clone = image_batch_tensor.clone()
        for image_tensor in image_batch_tensor_clone:
            features = np.array([self.__get_one_zero_rate(image_tensor), self.__get_height_width_rate(image_tensor), self.__get_top_bottom_nums_rate(image_tensor)])
            tensor = torch.from_numpy(features)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)
            features_tensor_list.append(tensor)
        stacked_tensor = torch.stack(features_tensor_list, dim=0)
        return stacked_tensor
