import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#bringing in OpenCV libraries
import cv2


# Read in the image and convert to grayscale
image = mpimg.imread('exit-ramp.jpg')
# 此处读取到的是RGB图像。若用cv2.imread('exit-ramp.jpg')读取到的图像不是RGB，而是BGR！
# image为ndarray类型，三维矩阵（行*列*通道），本图片为（540，960，3），
# 即为540行，960列，且i行j列的数据(一个像素点)为一个三元素向量[R,G,B]

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# cv2.cvtColor()变化不改变数据类型，仍为ndarray类型
# 但cv2.COLOR_RGB2GRAY使得三维数据变成了二维（Gray图像只有一个通道：灰度值）——（540，960）
# RGB图像中一个像素点包含一个三元素向量[R,G,B]，而Gray图像中一个像素点仅包含一个数据[G]
# 若是变化图像为cv2.COLOR_RGB2BGR，则得到图像依旧为三维数据（行*列*通道）(通道变成了BGR)

# plt.imshow(gray),plt.show()
# 此处若用plt.imshow(gray),plt.show()观察gray图片，会发现绿绿的，是因为matplotlib默认以彩色图显示图像，
# 不是RGB三通道的会自行渲染成RGB格式，可以在参数里设置：plt.imshow(,cmap=plt.cm.gray)，则可得灰度图


# Define a kernel size for Gaussian smoothing / blurring(高斯平滑函数)
# 通过平均来抑制噪声影响和虚假梯度(对图像做平滑处理，改善图像质量——更接近原图)
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
# cv2.GaussianBlur(src,kernel_size,sigmaX)
# 其中Gaussian kernel size(高斯核大小)分为宽和高，它们可以不同，但必须为正奇数；
# 或者，它们可以是零，然后由从西格玛计算而得。
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
# cv2.GaussianBlur()对灰度图像进行处理，得到的blur_gray仍为二维数据——（540，960）


# Define parameters for Canny and run it
# 灰度图像的像素值范围为0-255. 所以阈值范围为0-255
# John Canny本人推荐——低阈值:高阈值 = 1：2或1：3
# cv2.Canny()在内部有应用到高斯平滑，核大小为5*5
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# cv2.Canny()对高斯平滑后的灰度图像进行处理，得到的masked_edges仍为二维数据——（540，960）
# 得到边缘检测图像/梯度图
# plt.imshow(masked_edges),plt.show()
# 此处若用plt.imshow(masked_edges),plt.show()观察masked_edges图片，会发现紫紫的，matplotlib默认以彩图显示，
# 不是RGB三通道的会自行渲染成RGB格式，可以在参数里设置：plt.imshow(,cmap=plt.cm.Greys_r)，则可得灰度图

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image)*0 #creating a blank to draw lines on（全黑）
# line_image为三维数据——（540，960，3）

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
'''
# Hough变换函数一般都是对黑白二值图像(canny算法的输出)进行操作
# Hough变换函数为cv2.HoughLines()，但这种算法耗时耗力（尤其是计算每一个点(rho = 1)时），
# 即使经过了canny转换，有时点数依旧很庞大，这时我们采取一种概率挑选机制，不计算所有的点，
# 而是用随机的选取点来计算，相当于降采样。函数cv2.HoughLinesP()应运而生，是一种概率直线检测
# 由于cv2.HoughLinesP()是概率计算，所以选取的threshold较cv2.HoughLines()也应降低一些
# 而且cv2.HoughLinesP()的输出是直线端点坐标位（x1、y1、x2、y2）的数组
# 而cv2.HoughLines()还需复杂的转化才可得到此数据

# rho和theta为在Hough变换中的精度，一个点映射到Hough空间时绘出的离散函数的离散程度
# rho和theta越小，则精度越高，Hough空间的离散函数越趋于连续
# rho最小值为1(一个像素)，theta最小值为np.pi/180(一弧度)

# threshold为阈值，指定候选行能输出所需的最小投票数（给定网格单元中的交叉点中）
# 给定网格单元交叉点中，交叉线段条数大于threshold才被判断为一条直线

# 空的np.array([])只是一个占位符，不需要更改它。（不加np.array([])得到的lines的数据有所不同？？？）

# MinLineLengh为一条直线的最短长度，比这个短的都被忽略（以像素为单位）

# MaxLineGap为同一方向上两条线段判定为一条线段的最大允许间隔(断裂)，小于此值则把两条线段当成一条线段
# MaxLineGap值越大，允许线段上的断裂越大，越有可能检出潜在的直线段。
# 但如果MaxLineGap过大，会使得过多的不必要的直线段被连起来，使得满图都是直线
'''

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)  # cv2.line(图片，坐标1，坐标2，颜色，线宽)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
# np.dstack是堆栈数组按顺序深入（沿第三维）,又变成了一个（行*列*通道）的三维数据，
# i行j列的数据(一个像素点)为一个三元素向量[R,G,B],且由于堆栈的都是masked_edges，所以R=G=B
# 虽然这是一个RGB图像，但是R=G=B显示出的图像，看起来就像一个灰度图！


# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
# 妙!!增加权重，将color_edges透明度变为0.8，和line_image叠加起来构成图像。
# 若将0.8改为0.3则可看出函数作用，被叠加的两幅图像必须尺寸相同，类型相同
# dst(I)=saturate(src1(I)∗alpha+src2(I)∗beta+gamma)
# src1:第一个原数组； alpha:第一个数组元素权重； src2:第二个原数组； beta:第二个数组元素权重
# gamma:图1与图2作和后添加的数值。增大这个值，图片会变白，gamma=255以上就是纯白色了。
plt.subplot(224),plt.imshow(combo)

# Display the image
# camp='gray'和cmap='Greys_r'的区别：gray为灰度图，Greys_r为0通道/单通道的灰度图？？？
plt.subplot(221),plt.imshow(gray, cmap='gray')       #灰度图（Gray Image）
plt.subplot(222),plt.imshow(line_image, cmap=plt.cm.Greys_r)   #梯度图（Gradient Image）
plt.subplot(223),plt.imshow(color_edges)

plt.show()

