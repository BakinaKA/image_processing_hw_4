import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grayscale
def BGR2GRAY(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor Filter
def Gabor_filter(K_size=15, Sigma=7, Lambda=10,angle=0):
	d = K_size // 2

	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	for y in range(K_size):
		for x in range(K_size):
			px = x - d
			py = y - d

			theta = angle / 180. * np.pi
			_x = np.cos(theta) * px + np.sin(theta) * py
			gabor[y, x] = np.exp(-(px**2 + py**2) / (2 * Sigma**2)) * np.cos(2*np.pi*Lambda*_x)

	gabor /= np.sum(np.abs(gabor))
	return gabor


def Gabor_filtering(gray, K_size=15, Sigma=7, Lambda=10, angle=0):
    H, W = gray.shape
    out = np.zeros((H, W), dtype=np.float32)

    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Lambda=Lambda, angle=angle)
    plt.imshow(gabor)
    plt.show()
        
    out = cv2.filter2D(gray, -1, gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def Gabor_process(img):
    H, W, _ = img.shape

    gray = np.float32(BGR2GRAY(img))

    # define angle
    #As = [0, 90, 180]
    #As = [0, 45, 90, 135]
    As = [0,30,60,90,120,150]
    #As = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    for i, A in enumerate(As):
        _out = Gabor_filtering(gray, K_size=15, Sigma=0.9, Lambda=10.0, angle = A)
        
        plt.imshow( _out, cmap = 'gray')
        plt.show()

        out += _out
    
    out = out / out.max() * 255
    out = out.astype(np.uint8)
    for i in range(0,H):
        for j in range(0,W):
            if out[i,j]<127:
                out[i,j]=0
            else:
                out[i,j]=255
    
    return out

img = np.float32(cv2.imread("finger2.jpg"))

out = Gabor_process(img)
plt.imshow( out, cmap = 'gray')
plt.show()
cv2.imwrite("out_2_0.9.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()