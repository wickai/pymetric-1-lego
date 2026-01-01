# uv run python -m tools.metric.train_metric \
#     --cfg configs/nas/archive/imagenet/MbV2Nas_image_4card_600m_v23.yaml \
#     OUT_DIR ./output/nas_imagenet_v23.log \
#     PORT 12001


# uv run python -m tools.metric.train_metric \
#     --cfg configs/legonas/MbV2Nas_cifar2imagenet_4card_5blk.yaml \
#     OUT_DIR ./output/legonas/MbV2Nas_cifar2imagenet_4card_5blk_re \
#     PORT 12005
export PYMETRIC=`pwd`
export PYTHONPATH=`pwd`:$PYTHONPATH

uv run python3 tools/metric/train_metric.py \
    --cfg configs/nas/MbV2Nas_lw_imagenet_4card_5blk_exp02.yaml \
    OUT_DIR ./output/legonas/MbV2Nas_imagenet2imagenet_4card_5blk_exp02 \
    PORT 12002