
###############################################################################

# while

# tạo một list chứa các phần tử từ 1 đến 100
integers = []
x = 1
while x <= 100:
    integers += [x]
    x += 1

# tạo một list chứa 20 số hạng đầu tiên của dãy Fibonacci
a,b = 0,1
fibo = []
while len(fibo) < 20:
    a,b = b,a+b
    fibo += [a]

# tạo một list chứa 20 số chính phương đầu tiên
integers = []
x = 1
while x <= 20:
    integers += [x]
    x += 1

i = 0
squared_list = []
while len(squared_list) < 20:
    squared_num = integers[i] ** 2
    squared_list += [squared_num]
    i += 1

# Cho n=1, cần phải chia n cho 2 tối thiểu bao nhiêu lần để kết quả thu được
# nhỏ hơn 1/1000
n = 1
count = 0
print(n)
while n >= 1/10000:
    n /= 2
    count += 1
    print(n)

# Một trái phiếu trả lãi 8% một năm, cần nắm giữ bao lâu để nhân 3 tài sản
asset = 100
years = 0
while asset < 300:
    asset *= (1+0.08)
    years += 1

###############################################################################

