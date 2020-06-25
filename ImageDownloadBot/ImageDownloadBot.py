from bing_image_downloader import downloader

query_string_dog = "DogPictures"
query_string_cat = "CatPictures"
downloader.download(query_string_dog,limit = 100, adult_filter_off = True,force_replace=False)
downloader.download(query_string_cat,limit = 100, adult_filter_off = True,force_replace=False)


