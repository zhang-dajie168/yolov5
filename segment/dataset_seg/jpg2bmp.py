import os
from PIL import Image


def jpgToBmp(imgFile):
  dst_dir = "/home/ymt/yolov5/DataAugForObjectSegmentation/data2"

  for fileName in os.listdir(imgFile):
    if os.path.splitext(fileName)[1] == '.jpg' or os.path.splitext(fileName)[1] == '.png':
      name = os.path.splitext(fileName)[0]
      newFileName = name + ".bmp"

      img = Image.open(imgFile + "/" + fileName)
      img.save(dst_dir+"/"+newFileName)


def main():
  imgFile = "/home/ymt/yolov5/DataAugForObjectSegmentation/data3"

  jpgToBmp(imgFile)


if __name__ == '__main__':
   main()
