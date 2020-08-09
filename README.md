# Differentiable Elastic Transform based on Kornia

## Usage

Transform input image as follows.
```
import elastic_transform

et = elastic_transform.ElasticTransform(disp_scale=0.1, random_seed=42)
image_transformed = et.forward(image)
```

## Example
Run `python example.py`, it will give the following output.

