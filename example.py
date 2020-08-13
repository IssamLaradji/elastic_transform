import torchvision
import elastic_transform
import numpy as np, time

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

    fname_out = 'transformed.jpg'
    image_pil = torchvision.transforms.ToPILImage()(image_transformed.cpu().squeeze())
    image_pil.save(fname_out)

    org = (image.cpu().numpy().squeeze()*255).astype('uint8')
    out = (image_transformed.detach().cpu().numpy().squeeze()*255).astype('uint8')

    output = np.concatenate([org, out, ],axis=1)
    Image.fromarray(output).save(fname_out)

    print('%.3f seconds with %s to process image "%s", '  
          'saved as "%s"' % ( e_time, device, fname, fname_out))