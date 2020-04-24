import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['stiching', 'end2end', 'end2end+retrain', 'Face search at scale', 'Masi et al', 'PAMs']    # 横坐标刻度显示值
num_list1 = [0.881, 0.891, 0.889, 0.729, 0.886, 0.826]      # 纵坐标值1
sd1 = [0.018, 0.016, 0.013, 0.035, 0.017, 0.018]
num_list2 = [0.714, 0.764, 0.762, 0.510, 0.725, 0.652]      # 纵坐标值2
sd2 = [0.034, 0.031, 0.033, 0.061, 0.044, 0.037]
num_list3 = [0.913, 0.924, 0.920, 0.822, 0.906, 0.840]      # 纵坐标值3
sd3 = [0.013, 0.016, 0.015, 0.023, 0.013, 0.012]

# x = range(len(num_list1))
x = [0, 2, 4, 6, 8, 10]
rects1 = plt.bar(x, height=num_list1, width=0.6, yerr=sd1, error_kw = {'ecolor' : '0.2', 'capsize' :6}, alpha=0.8, color='red', label="TAR@FAR=0.01")
rects2 = plt.bar([i + 0.6 for i in x], height=num_list2, width=0.6, yerr=sd2, error_kw = {'ecolor' : '0.2', 'capsize' :6}, color='green', label="TAR@FAR=0.001")
rects3 = plt.bar([i + 1.2 for i in x], height=num_list3, width=0.6, yerr=sd3, error_kw = {'ecolor' : '0.2', 'capsize' :6}, color='blue', label="Rank-1")

plt.ylim(0, 1.2)     # y轴取值范围
plt.ylabel("Accuracy")
# """
# 设置x轴刻度显示值
# 参数一：中点坐标
# 参数二：显示值
# """
plt.xticks([index + 0.4 for index in x], label_list)
plt.xlabel("Mothed")
plt.title("Compare")
plt.legend()     # 设置题注
plt.show()

# from matplotlib import pyplot
# import matplotlib.pyplot as plt
 
# # names = range(8,21)
# # names = [str(x) for x in list(names)]
# names = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
 
# x = range(len(names))
# y_train = [2.46, 5.75, 6.61, 7.34, 7.92, 8.57, 9.05, 9.43, 9.86, 10.29]
# y_test  = [2.79, 6.54, 7.93, 8.52, 9.14, 9.74, 10.08, 10.67, 12.59, 13.81]
# #plt.plot(x, y, 'ro-')
# #plt.plot(x, y1, 'bo-')
# #pl.xlim(-1, 11)  # 限定横轴的范围
# #pl.ylim(-1, 110)  # 限定纵轴的范围
 
 
# plt.plot(x, y_train, marker='o', mec='r', mfc='w',label='EPNP')
# plt.plot(x, y_test, marker='*', ms=10,label='POSIT')
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=1)
 
# plt.margins(0)
# plt.subplots_adjust(bottom=0.10)
# plt.ylim(0, 20)
# plt.xlabel('Angle') #X轴标签
# plt.ylabel("Error") #Y轴标签
# # pyplot.yticks([0.750,0.800,0.850])
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib
# # 设置中文字体和负号正常显示
# # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # matplotlib.rcParams['axes.unicode_minus'] = False

# label_list = ['10', '20', '40']    # 横坐标刻度显示值
# num_list1 = [5.78, 11.81, 24.37]      # 纵坐标值1
# num_list2 = [0.66, 1.42, 2.95]      # 纵坐标值2
# x = range(len(num_list1))
# """
# 绘制条形图
# left:长条形中点横坐标
# height:长条形高度
# width:长条形宽度，默认值0.8
# label:为后面设置legend准备
# """
# rects1 = plt.bar(x, height=num_list1, width=0.4, alpha=0.8, color='yellow', label="EPNP")
# rects2 = plt.bar([i + 0.4 for i in x], height=num_list2, width=0.4, color='blue', label="POSIT")
# plt.ylim(0, 25)     # y轴取值范围
# plt.ylabel("Time")
# """
# 设置x轴刻度显示值
# 参数一：中点坐标
# 参数二：显示值
# """
# plt.xticks([index + 0.2 for index in x], label_list)
# plt.xlabel("Number")
# plt.title("Compare")
# plt.legend()     # 设置题注
# # 编辑文本
# # for rect in rects1:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
# # for rect in rects2:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
# plt.show()