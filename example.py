import torchvision
import elastic_transform
import elastic_transform_numpy
import numpy as np 

from PIL import Image

fname = 'lungs.jpg'

# image_pil = Image.open(fname).resize((10,10))
image_pil = Image.open(fname)
image = torchvision.transforms.ToTensor()(image_pil)[None][:,[0]]

et = elastic_transform.ElasticTransform(alpha=1,  sigma=12, random_seed=42)
image_transformed = et.forward(image)

image_pil = torchvision.transforms.ToPILImage()(image_transformed.squeeze())
image_pil.save('lungs_transformed.jpg')

image_transformed_numpy = elastic_transform_numpy.elastic_transform_numpy(image.numpy().squeeze(), 
                                alpha=1, sigma=12, random_state=42)

org = (image.numpy().squeeze()*255).astype('uint8')
out = (image_transformed.detach().numpy().squeeze()*255).astype('uint8')
out_numpy = (image_transformed_numpy*255).astype('uint8')

output = np.concatenate([org, out_numpy, out, ],axis=1)
Image.fromarray(output).save('lungs_transformed.jpg')