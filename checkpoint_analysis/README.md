# Generating heat map for a checkpoint


First get all the tensor names:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt --all_tensor_names
```

Run weight_heatmap.py:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt --tensor_name=<tensor_name> --mask=mask_name

$ python weight_heatmap.py --file_name=vgg16_ckpt/vgg_check/model.ckpt-3947 --tensor_name=vgg_16/conv1/conv1_1/weights --mask=vgg_16/conv1/conv1_1/mask

```

To get shapes of tensors:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt
```
