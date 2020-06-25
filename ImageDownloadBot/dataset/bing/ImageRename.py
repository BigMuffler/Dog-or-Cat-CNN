import os
pathdog="D:/Personal/DeepLearning/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Section 40 - Convolutional Neural Networks (CNN)/ImageDownloadBot/dataset/bing/DogPictures/"
pathcat = "D:/Personal/DeepLearning/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Section 40 - Convolutional Neural Networks (CNN)/ImageDownloadBot/dataset/bing/CatPictures/"
i = 1
for filename in os.listdir(pathdog):
      my_dest ="dog." + str(i+4000) + ".jpg"
      my_source =pathdog + filename
      my_dest =pathdog + my_dest

      os.rename(my_source, my_dest)
      i += 1
i = 1
for filename in os.listdir(pathcat):
      my_dest ="dog." + str(i+4000) + ".jpg"
      my_source =pathcat + filename
      my_dest =pathcat + my_dest

      os.rename(my_source, my_dest)
      i += 1