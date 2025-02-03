import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
from skimage import color

def evaluate(model, dataset, final=False):
    print('Model performance:')
    
    train_score = model.evaluate(dataset.x_train_gray, dataset.x_train, verbose=False)
    print('\tTrain loss:          %0.4f' % train_score[0])
    
    if final:
        test_score = model.evaluate(dataset.x_test_gray, dataset.x_test, verbose=False)
        print('\tTest loss:           %0.4f' % test_score[0])
    else:
        val_score = model.evaluate(dataset.x_valid_gray, dataset.x_valid, verbose=False)
        print('\tValidation loss:     %0.4f' % val_score[0])
    
    return train_score

def pred_test(model, dataset, name):
    logits = model.predict(dataset.x_test)
    pred = np.argmax(logits, axis=1)
    df = pd.DataFrame({'class': pred})
    df.index.name = 'id'
    df.to_csv(name)
    print('Done!Please upload your file to Kaggle!')
    return pred

def compute_ssim(image1, image2):

    ssim_score = ssim(image1, image2, data_range=image1.max() - image1.min(), channel_axis=2, win_size=11)
    return ssim_score

def compute_snr(image1, image2):

    image1 = image1 / image1.max()
    image2 = image2 / image2.max()
    
    signal_power = np.mean(image1 ** 2)
    noise_power = np.mean((image1 - image2) ** 2)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))  
    
    return snr

def compute_scielab(image1, image2):

    lab1 = color.rgb2lab(image1)
    lab2 = color.rgb2lab(image2)

    deltaE = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    
    csf = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  
    deltaE_filtered = convolve(deltaE, csf, mode='reflect')
    
    scielab_score = np.mean(deltaE_filtered)
    return scielab_score