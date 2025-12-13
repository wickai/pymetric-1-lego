from .mbv2_searchspace import MobileNetSearchSpace
from metric.core.config import cfg


def mobilenet_v2_nas():
    search_space = MobileNetSearchSpace(
        num_classes=1000,
        small_input=False
    )
    # best_individual = {'op_codes': [
    # 7, 7, 10, 6, 7, 6, 6, 10, 5, 3, 3, 3, 2, 10, 7, 10, 2], 'width_codes': [1, 1, 1, 1, 1, 0, 1]}
    # 7.38m
    # best_individual = {'op_codes': [
    #     10, 2, 3, 10, 11, 11, 7, 10, 11, 8, 5, 4, 10, 8, 10, 8, 7], 'width_codes': [1, 1, 1, 1, 1, 0, 0]}
    # 7.23m
    # best_individual = {'op_codes': [10, 7, 7, 7, 3, 11, 6, 10, 8, 6,
    #                                 0, 12, 7, 1, 9, 12, 3], 'width_codes': [1, 1, 1, 1, 1, 0, 0]}
    model = search_space.get_model(
        cfg.MODEL.OP_CODES,  cfg.MODEL.WIDTH_CODES)
    return model
