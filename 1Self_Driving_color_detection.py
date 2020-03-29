import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
# RGB图片通过image.shape来转换为数组，就变成了一个rows*cols*channels（行*列*通道）的三维矩阵
# 因为是RGB，所以是三通道，且RGB按位置分别为通道[0][1][2]，此功能可用于访问像素值
image = mpimg.imread('test.jpg')   #读取图像
'''
此处读取到的是RGB格式图像，若用opencv库的cv2.imread('test.jpg')，则得到BGR图像
RGB与BGR的不同与下方阈值设置部分代码有关，不同格式位置不同
'''
print('This image is: ',type(image), 'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection 
# overlaid on the original.
# 复制数组中某个位置的值时，不会指向同一个地址，只是单纯的复制数值
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
color_select = np.copy(image)
line_image = np.copy(image)



''' 颜色选择 '''
# Define our color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
# \ 表示换行（在程序中换行）      | 表示按位或运算（补充： ^ 表示按位异或）
# 通过 : 来遍历所有像素点，0，1，2分别表示R、G、B
color_thresholds =   (image[:,:,0] < rgb_threshold[0]) \
                   | (image[:,:,1] < rgb_threshold[1]) \
                   | (image[:,:,2] < rgb_threshold[2])



''' 区域选择 '''
# 在图像处理中，原点（x=0，y=0）在左上角。
# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
left_bottom = [0, 540]
right_bottom = [960, 540]
apex = [480, 320]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit() returns the coefficients [A, B] of the fit
# np.ployfit(x坐标，y坐标，n)    x,y坐标为多个，用（）括起来，n为拟合阶数
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
# np.meshgrid(x,y)用于构建坐标矩阵XX,YY，其中x，y是网格点的横纵坐标列向量
# XX的第n行与YY的第n行对应，构成坐标，即(XX[i][j],YY[i][j])构成所有网格点
# np.arange(a,b,m)  a为起点，b为终点（不可达），m为步长（默认为1，m可为小数），生成一个ndarray类型数据
# python中类似的函数为range(a,b,m)，此处m不可为小数（默认为1）
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))

# Mask color and region selection
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
# Color pixels red where both color and region selections met
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]
# 用print(color_thresholds)看起来像一个多维数组的数据，但用type查询可得类型为<class 'numpy.ndarray'>
# 返回值为True or False。上面两条语句的用法，仅在<class 'numpy.ndarray'>可用，在数组类型中不可使用！！


# Display the image and show region and color selections
# plt.show()只显示最近的plt.imshow()，所以用plt.subplot()放置在不同位置
plt.subplot(2,2,1),plt.imshow(image)
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2),plt.imshow(color_select)
# 画线，坐标的写法留意一下
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]    
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, 'b--', lw=4)   #line wide = 4
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3),plt.imshow(line_image)
plt.xticks([]),plt.yticks([])

plt.show()
