import matplotlib.pyplot as plt
import os 
import cv2 as cv
from src.model import SiameseModel
from src.utils import distance
import torch
from torch import nn

def main():

    # loading model
    weight = torch.load('./checkpoints-saved/checkpoint_exp3_100.pkl')
    model = SiameseModel()
    model = nn.DataParallel(model)
    model = model.to('cuda')

    model.load_state_dict(weight['model'])

    # testing
    anchor = cv.imread('../utils/images-demo/siamese-test/anchor/0028_c1s4_033506_01.jpg', cv.IMREAD_ANYCOLOR)
    anchor = cv.cvtColor(anchor, cv.COLOR_BGR2RGB).reshape(1, *anchor.shape)
    tensor_anchor = torch.from_numpy(anchor).permute(0,3,1,2)

    images = os.listdir('../utils/images-demo/siamese-test/other')
    
    fig, axes = plt.subplots(nrows=len(images), ncols=3)
    fig.tight_layout()

    axes[0][0].set_title('anchor')
    axes[0][1].set_title('other')
    axes[0][2].set_title('distance')

    for row, ax in enumerate(axes):
        cmp_image = cv.imread(os.path.join('../utils/images-demo/siamese-test/other/',images[row]), cv.IMREAD_ANYCOLOR)
        cmp_image = cv.cvtColor(cmp_image, cv.COLOR_BGR2RGB).reshape(1, *cmp_image.shape)
        tensor_cmp_image = torch.from_numpy(cmp_image).permute(0,3,1,2)
    
        x, y, _ = model(tensor_anchor/255, tensor_cmp_image/255, tensor_anchor/255)

        ax[0].imshow(anchor[0])
        ax[0].axis('off')

        ax[1].imshow(cmp_image[0])
        ax[1].axis('off')

        ax[2].text(0,0, '%.2f'%distance(x,y))
        ax[2].axis('off')
        
    plt.savefig(f"./siamese-test.png", dpi=1200)
    plt.close()
        
if __name__ == '__main__':
    main()