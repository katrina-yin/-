# 算术运算
print(2 + 2)
print(50 - 5 * 6)
print(8 / 5)  # 浮点数除法
print(8 // 5)  # 整数除法

# 字符串操作
print('"I\'m ZRen," he said.')
s = 'First word.\nSecond word.'
print(s)
print('School' ' of' ' Software')  # 字符串拼接，限于常量，注意空格
print('b' + 2 * 'an' + 'a')  # 字符串重复

programming = 'Programming'
print(programming[0])  # 下标
print(programming[-1])
print(programming[5])
print(programming[:2])  # 切片
print(programming[4:])
print(programming[-2:])
print(programming[:2] + programming[2:])

# 列表操作
square = [1, 4, 9, 16, 25]
print(square)
print(square[0])
print(square[-1])
print(square[-3:])
print(square[:])
print(square + square)
print(square * 3)
square[2] = 0  # 列表元素可变
print(square)
square.append(-1)
print(square)
print(len(square))  # 返回列表长度
a = [1, 2, 3]
b = [4, 5, 6]
c = [a, b]  # 列表嵌套
print(c)
print(c[0][1])

# 条件判断
x = 20
if x < 0:
    x = 0
    print("negative value of x")
elif x == 0:
    print("zero")
elif x == 1:
    print("single")
else:
    print("More")

# 循环
words = ['Java', 'C++', 'Ruby']
for w in words:
    print(w, len(w))

for i in range(5):
    print(i)

# 函数
def foo():
    print("hello world")

def add(x, y):
    return x + y

print(add(3, 4))
foo()

# 基于 turtle 的图形绘制
import turtle

# 绘制正方形
def square(length):
    for _ in range(4):
        turtle.forward(length)
        turtle.left(90)

# 绘制分形树
def tree(len, n):
    if n <= 0:
        return
    else:
        turtle.forward(len)
        turtle.left(45)
        tree(len * 0.5, n - 1)
        turtle.right(90)
        tree(len * 0.5, n - 1)
        turtle.left(45)
        turtle.backward(len)

# 绘制雪花
def snowflake(length, threshold):
    if length < threshold:
        turtle.forward(length)
    else:
        length /= 3
        # Segment 1
        snowflake(length, threshold)
        turtle.left(60)
        # Segment 2
        snowflake(length, threshold)
        turtle.right(120)
        # Segment 3
        snowflake(length, threshold)
        turtle.left(60)
        # Segment 4
        snowflake(length, threshold)

# 绘制分形三角形
def triangle(length, threshold):
    if length < threshold:
        return
    else:
        for i in range(3):
            turtle.forward(length)
            triangle(length / 2, threshold)
            turtle.backward(length)
            turtle.left(120)

# 启动图形绘制
turtle.speed(0)  # 设置速度为最快
# 绘制树
tree(200, 9)

# 绘制雪花
turtle.reset()  # 清空画布
for i in range(3):
    snowflake(300, 10)
    turtle.right(120)

# 绘制三角形
turtle.reset()  # 清空画布
triangle(100, 1)

# 关闭图形窗口
turtle.done()
