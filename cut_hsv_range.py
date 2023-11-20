python

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.imagecopy = self.image.copy()
        self.list1 = []
        self.list2 = []
        self.num = 0

    def resize_image(self):
        self.image = cv2.resize(self.image, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def getpos(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            HSV = self.HSV2.copy()
            cv2.line(HSV, (0, y), (HSV.shape[1]-1, y), (255, 255, 255), 1, 4)
            cv2.line(HSV, (x, 0), (x, HSV.shape[0] - 1), (255, 255, 255), 1, 4)
            cv2.imshow("imageHSV", HSV)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.num += 1
            HSV = self.HSV2.copy()
            if self.num == 1:
                self.list1.append([x, y])
                print('请点击HSV图片上第二个点')
            if self.num == 2:
                self.num = 0
                self.list2.append([x, y])
                hlist = []
                slist = []
                vlist = []
                for i in range(min(self.list1[-1][0], self.list2[-1][0]), max(self.list1[-1][0], self.list2[-1][0])):
                    for j in range(min(self.list1[-1][1], self.list2[-1][1]), max(self.list1[-1][1], self.list2[-1][1])):
                        hlist.append(self.HSV[j, i][0])
                        slist.append(self.HSV[j, i][1])
                        vlist.append(self.HSV[j, i][2])
                hlist.sort()
                slist.sort()
                vlist.sort()
                print(hlist)
                print(slist)
                print(vlist)
                print('请点击HSV图片上第一个点')
                print((hlist[0], slist[0], vlist[0]), (hlist[-1], slist[-1], vlist[-1]))

    def process_image(self):
        self.resize_image()
        self.HSV = self.image.copy()
        self.HSV2 = self.image.copy()
        cv2.imshow("imageHSV", self.HSV)
        cv2.setMouseCallback("imageHSV", self.getpos)
        cv2.waitKey(0)

