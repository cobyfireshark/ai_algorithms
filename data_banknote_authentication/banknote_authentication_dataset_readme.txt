Data were extracted from images that were taken from genuine and forged banknote-like specimens.  For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

https://archive.ics.uci.edu/dataset/267/banknote+authentication

List of the five variables in the dataset:
    variance of Wavelet Transformed image (continuous)
    skewness of Wavelet Transformed image (continuous)
    kurtosis of Wavelet Transformed image (continuous)
    entropy of image (continuous)
    class (integer)

In the banknote dataset, if the classes are evenly split between authentic and inauthentic banknotes, the Zero Rule Algorithm will have 50% accuracy. This is because it always predicts the most common class; if the classes are balanced, it will be correct half the time and incorrect the other half.