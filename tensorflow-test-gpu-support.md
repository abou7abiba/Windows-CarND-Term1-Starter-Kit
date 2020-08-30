```python
# importing the tensorflow package
import tensorflow as tf 
```


```python
tf.test.is_built_with_cuda()
```




    True




```python
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
```




    True




```python
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




```python

```
