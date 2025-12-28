# uv run python -m tools.metric.train_metric \
#     --cfg configs/nas/archive/imagenet/MbV2Nas_image_4card_600m_v23.yaml \
#     OUT_DIR ./output/nas_imagenet_v23.log \
#     PORT 12001


uv run python -m tools.metric.train_metric \
    --cfg configs/legonas/MbV2Nas_cifar2imagenet_4card_5blk.yaml \
    OUT_DIR ./output/legonas/MbV2Nas_cifar2imagenet_4card_5blk_re \
    PORT 12005