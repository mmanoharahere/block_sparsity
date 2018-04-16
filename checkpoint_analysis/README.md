# Generating heat map for a checkpoint


First get all the tensor names:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt --all_tensor_names
```

Run weight_heatmap.py:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt --tensor_name=<tensor_name>
```

To get shapes of tensors:

```shell
$ python weight_heatmap.py --file_name=<checkpoint_name>.ckpt
```
