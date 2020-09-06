import torchvision
import elastic_transform
import numpy as np, time
import torch

from PIL import Image

if __name__ == "__main__":
    fname = 'base/lungs.jpg'
    device = 'cpu'

    # image_pil = Image.open(fname).resize((10,10))
    image_pil = Image.open(fname)
    
    image = torchvision.transforms.ToTensor()(image_pil)[None][:,[0]].to(device)

    s_time = time.time()
    et = elastic_transform.ElasticTransform(alpha=1,  
                        sigma=12, random_seed=42).to(device)
    image_transformed = et.forward(image)
    e_time = time.time() - s_time

    img_cat = torch.cat([image, image_transformed], dim=0)
    output = (torchvision.utils.make_grid(img_cat).detach().permute(1,2,0).cpu().numpy().squeeze()*255).astype('uint8')
    fname_out = 'transformed.jpg'
    Image.fromarray(output).save(fname_out)

    print('%.3f seconds with %s to process image "%s", '  
          'saved as "%s"' % ( e_time, device, fname, fname_out))